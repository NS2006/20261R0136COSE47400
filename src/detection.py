import matplotlib.pyplot as plt
import os 
import cv2
import numpy as np
from tqdm import tqdm
import faiss
import tifffile as tiff
import time
import torch
import json
from sklearn.cluster import KMeans
from src.utils import augment_image, dists2map, min_max_norm, cvt2heatmap, heatmap_on_image
from src.post_eval import mean_top1p
from src.sampler import GreedyCoresetSampler

def fill_closed_regions(image):
    if image is None:
        print("Invalid input image")
        return None

    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filled = np.zeros_like(image)
    cv2.drawContours(filled, contours, contourIdx=-1, color=255, thickness=cv2.FILLED)

    return filled

def compute_high_res_mask(img_rgb, grid_size):
    """
    Extracts a precise mask for transparent or challenging objects (e.g., Vials) at high resolution, 
    then downsamples it to match the DINOv2 patch grid size (e.g., 16x16).
    This prevents the loss of fine structural details that occurs if the image is resized directly.
    """
    # 1. Grayscale conversion and Otsu binarization
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 2. High-resolution morphological operations (filling internal holes/gaps)s.
    kernel = np.ones((5, 5), np.uint8)
    closed_mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 3. Downsample the cleaned mask to the DINOv2 patch grid size
    mask_resized = cv2.resize(closed_mask, (grid_size[1], grid_size[0]), interpolation=cv2.INTER_AREA)
    
    # 4. Re-binarize the resized mask into a Boolean format (True/False) and flatten it into a 1D array
    final_mask = (mask_resized > 127).flatten()
    
    return final_mask

def run_anomaly_detection_multilayer(
        model,
        object_name,
        data_root,
        n_ref_samples,
        object_anomalies,
        plots_dir,
        device,
        save_examples=False,
        masking=None,
        mask_ref_images=False,
        rotation=False,
        knn_metric='L2_normalized',
        knn_neighbors=1,
        faiss_on_cpu=False,
        seed=0,
        save_patch_dists=True,
        save_tiffs=False):
    """
    Updated to support multi-layer feature extraction and layer-wise knn matching.
    """

    assert knn_metric in ["L2", "L2_normalized"]
    type_anomalies = list(set(object_anomalies[object_name] + ['good']))

    img_ref_folder = f"{data_root}/{object_name}/train/good/"
    img_ref_samples = sorted(os.listdir(img_ref_folder))

    ####################################### Coreset Selection ######################################## 

    # 1. Global Feature Extraction (CLS Tokens)
    cls_features = []
    valid_img_names = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with torch.inference_mode():
        for img_name in tqdm(img_ref_samples, desc="Extracting CLS features", leave=False):
            image_path = os.path.join(img_ref_folder, img_name)
            img_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            img_tensor, _ = model.prepare_image(img_rgb)
            
            # Extract the global semantic representation (CLS token)
            cls_feats = model.extract_cls_features(img_tensor)
            cls_features.append(cls_feats.squeeze().cpu())
            valid_img_names.append(img_name)

    cls_features = torch.stack(cls_features).to(device)  

    # 2. Dynamic Ratio Calculation
    n_ref_samples_selection = 100 
    target_percentage = min(1.0, n_ref_samples_selection / len(cls_features))
    
    # 3. Greedy Coreset Sub-sampling
    sampler = GreedyCoresetSampler(percentage=target_percentage, device=device, dimension_to_project_features_to=1024)
    selected_indices = sampler.run(cls_features)
    
    # 4. Final Reference Image Isolation
    img_ref_samples = [valid_img_names[idx] for idx in selected_indices][:n_ref_samples_selection]

    print(f"Final number of selected images for patch extraction: {len(img_ref_samples)}\n")
    ##################################################################################################
    


    ####################################################################################
    # Multi-Layer Feature Extraction & Dual Memory Bank Initialization
    # ----------------------------------------------------------------------------------
    # Instead of using a single memory bank like the SuperAD baseline, 
    # we initialize two separate dictionaries to hold multi-layer features.
    feature_refs = {}     # Foreground Memory Bank (Normal Object Patches)
    bg_feature_refs = {}  # Background Memory Bank (Noise/Background Patches)
    grid_size = None

    with torch.inference_mode():
        start_time = time.time()

        for img_name in tqdm(img_ref_samples, desc="Extracting reference features", leave=False):
            image_path = os.path.join(img_ref_folder, img_name)
            img_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

            # Apply rotation augmentations if enabled, otherwise process the original image
            aug_images = augment_image(img_rgb) if rotation else [img_rgb]
            for aug in aug_images:
                img_tensor, grid_size = model.prepare_image(aug)
                
                # Extract features from the 4 designated DINOv2 layers
                feats_dict = model.extract_features(img_tensor)
                idx = 0

                # ----------------------------------------------------------------------
                # Adaptive Masking Strategy Based on Object Class
                # ----------------------------------------------------------------------
                if object_name in ['fabric', 'rice', 'sheet_metal']:
                    # Textures span the entire image; there is no "background."
                    # We set the mask to all True (1s) so every patch goes to the Object Bank.
                    total_patches = grid_size[0] * grid_size[1] if isinstance(grid_size, (list, tuple)) else grid_size * grid_size
                    mask = np.ones(total_patches, dtype=bool)
                
                elif object_name in ['vial', 'fruit_jelly']:
                    # Transparent or semi-transparent objects use our custom high-res morphological mask
                    # to prevent background bleeding into the object memory bank.
                    mask = compute_high_res_mask(aug if 'aug' in locals() else img_rgb, grid_size)
                
                else:
                    # Standard isolated objects (e.g., cans, walnuts) use the default DINOv2 feature masking
                    mask = model.compute_background_mask(feats_dict[0], grid_size, threshold=1, masking_type=masking)
                
                mask_bg = ~mask

                # ----------------------------------------------------------------------
                # Feature Routing & Separation (Per Layer)
                # ----------------------------------------------------------------------
                for idx, feats in enumerate(feats_dict):
                    layer_key = f'layer{idx}'
                    
                    if layer_key not in feature_refs:
                        feature_refs[layer_key] = []
                        bg_feature_refs[layer_key] = []

                    feature_refs[layer_key].append(feats[mask])
                    bg_feature_refs[layer_key].append(feats[mask_bg])
    ####################################################################################



        ####################################################################################
    # PHASE 2b: SVD Basis Construction & Background Noise Suppression
    # ----------------------------------------------------------------------------------
    # We dynamically calculate the principal components (axes of variation)
    # for both the background (to identify noise like lighting/glare) and the foreground.
    ####################################################################################
    P_bg_banks = {}     # Primary Noise/Illumination Suppression Filter (Projection Matrix)
    U_normal_banks = {} # Secondary Normal Object SVD Basis
    
    alpha = 0.5         
    variance_threshold = 0.85

    for layer_name in feature_refs.keys():
        bg_feats = torch.from_numpy(np.concatenate(bg_feature_refs[layer_name])).to(device)
        norm_feats_raw = torch.from_numpy(np.concatenate(feature_refs[layer_name])).to(device)
        feat_dim = norm_feats_raw.shape[1]

        if bg_feats.shape[0] == 0:
            # [Texture Data] If there is no background (e.g., fabric, wood), the noise filter 
            # simply becomes an Identity Matrix (allowing 100% feature passthrough).
            P_bg = torch.eye(feat_dim, device=device)
            k_bg = 0
        else:
            ################################################################################
            # [Normal/Transparent Objects] Extracting axes of noise from the background
            ################################################################################
            # Center the background features to extract pure variance
            bg_mean = torch.mean(bg_feats, dim=0, keepdim=True)
            bg_feats_centered = bg_feats - bg_mean
            
            # --- Covariance Trick instead of massive SVD ---
            X_b = bg_feats_centered.T
            C_b = torch.mm(X_b, X_b.T) 
            L_b, U_b = torch.linalg.eigh(C_b) 
            
            L_b = L_b.flip(dims=(0,))
            U_b = U_b.flip(dims=(1,))
            L_b = torch.clamp(L_b, min=0)
            
            explained_variance_b = L_b / torch.sum(L_b)
            k_bg = torch.searchsorted(torch.cumsum(explained_variance_b, dim=0), variance_threshold).item() + 1
            U_bg_k = U_b[:, :k_bg] # Truncate to the top k principal components
            
            # Construct the Soft Projection Matrix to subtract background noise from the feature space
            I = torch.eye(feat_dim, device=device)
            P_bg = I - alpha * torch.mm(U_bg_k, U_bg_k.T)
            ################################################################################

        # Apply the projection matrix to purify the normal object features
        P_bg_banks[layer_name] = P_bg
        norm_feats_purified = torch.mm(norm_feats_raw, P_bg.T)
        
        X_n = norm_feats_purified.T 
        C_n = torch.mm(X_n, X_n.T) 
        L_n, U_n = torch.linalg.eigh(C_n)
        
        L_n = L_n.flip(dims=(0,))
        U_n = U_n.flip(dims=(1,))
        L_n = torch.clamp(L_n, min=0)
        
        explained_variance_n = L_n / torch.sum(L_n)
        k_norm = torch.searchsorted(torch.cumsum(explained_variance_n, dim=0), variance_threshold).item() + 1
        
        U_normal_banks[layer_name] = U_n[:, :k_norm]
        print(f"[{layer_name}] k_bg: {k_bg}, k_norm: {k_norm}")
    ####################################################################################
        



        time_memorybank = time.time() - start_time

        inference_times = {}
        anomaly_scores = {}

        for anomaly_type in tqdm(type_anomalies, desc=f"Processing {object_name}"):
            test_dir = f"{data_root}/{object_name}/test_public/{anomaly_type}"
            os.makedirs(f"{plots_dir}/anomaly_maps/seed={seed}/{object_name}/test/{anomaly_type}", exist_ok=True)
            os.makedirs(f"{plots_dir}/anomaly_maps/seed={seed}/{object_name}/test_hm_on_img/{anomaly_type}", exist_ok=True)

            for idx, test_img_name in enumerate(sorted(os.listdir(test_dir))):
                start_time = time.time()
                test_path = os.path.join(test_dir, test_img_name)
                
                img_rgb = cv2.cvtColor(cv2.imread(test_path), cv2.COLOR_BGR2RGB)
                img_tensor, _ = model.prepare_image(img_rgb)
                feats_dict = model.extract_features(img_tensor)

                dists_per_layer = []
                
                # Grouping
                if object_name in ['fabric', 'rice', 'sheet_metal']:
                    total_patches = grid_size[0] * grid_size[1] if isinstance(grid_size, (list, tuple)) else grid_size * grid_size
                    mask = np.ones(total_patches, dtype=bool)
                elif object_name in ['vial', 'fruit_jelly']: 
                    mask = compute_high_res_mask(img_rgb, grid_size)
                else:
                    mask = model.compute_background_mask(feats_dict[0], grid_size, threshold=1, masking_type=masking)


                for num, feats in enumerate(feats_dict):
                    masked_feats = feats[mask]

                    ###################################################################################
                    # Soft Projection & Residual-Based Anomaly Scoring
                    # ---------------------------------------------------------------------------------
                    # Instead of using standard K-NN distance to a memory bank, 
                    # we evaluate test patches by projecting them through our pre-computed SVD bases 
                    # and measuring the unexplained residual variance.
                    ###################################################################################
                    y = torch.from_numpy(masked_feats).to(device)
                    layer_name = f'layer{num}'
                    
                    # 1. Background Noise Suppression (Soft Projection)
                    P_bg = P_bg_banks[layer_name]
                    y_no_bg = torch.mm(y, P_bg.T) 
                    
                    # 2. Normal Subspace Projection
                    U_norm = U_normal_banks[layer_name]
                    P_norm = torch.mm(U_norm, U_norm.T)
                    y_proj = torch.mm(y_no_bg, P_norm.T)
                    
                    # 3. Residual Extraction (The True Anomaly)
                    y_residual = y_no_bg - y_proj 
                    
                    # 4. Final Anomaly Scoring
                    dists = torch.sum(y_residual ** 2, dim=1).cpu().numpy()
                    ###################################################################################




                    dmap = np.zeros_like(mask, dtype=float)
                    dmap[mask] = dists.squeeze()
                    dmap = dmap.reshape(grid_size)

                    dmap_resized = cv2.resize(dmap, (img_rgb.shape[1], img_rgb.shape[0]))
                    dists_per_layer.append(dmap_resized)

                ###################################################################################
                # Multi-Layer Heatmap Fusion (Weighted Averaging)
                # ---------------------------------------------------------------------------------
                # Instead of averaging all 4 DINOv2 layers equally, we apply 
                # a custom weighted average. Lower layers (0 and 1) primarily capture low-level 
                # textures and edges, which are highly susceptible to noise. Deeper layers (2 and 3) 
                # provide robust, semantic object-level features. 
                ###################################################################################
                
                # Optimal layer weights derived from our empirical experiments (Sums to 1.0)
                # [Layer 0: 10%, Layer 1: 18%, Layer 2: 32%, Layer 3: 40%]
                weights = [0.10, 0.18, 0.32, 0.40]  
                anomaly_map = np.average(dists_per_layer, axis=0, weights=weights)
                

                anomaly_map_norm = min_max_norm(anomaly_map)
                score = mean_top1p(anomaly_map.flatten())

                inference_times[f"{anomaly_type}/{test_img_name}"] = time.time() - start_time
                anomaly_scores[f"{anomaly_type}/{test_img_name}"] = score

                if save_tiffs:
                    heatmap = cvt2heatmap(anomaly_map_norm * 255)
                    hm_on_img = heatmap_on_image(heatmap, img_rgb)
                    fname = os.path.splitext(test_img_name)[0]
                    cv2.imwrite(f"{plots_dir}/anomaly_maps/seed={seed}/{object_name}/test_hm_on_img/{anomaly_type}/{fname}.jpg", hm_on_img)
                    tiff.imwrite(f"{plots_dir}/anomaly_maps/seed={seed}/{object_name}/test/{anomaly_type}/{fname}.tiff", anomaly_map)
                if save_patch_dists:
                    np.save(f"{plots_dir}/anomaly_maps/seed={seed}/{object_name}/test/{anomaly_type}/{test_img_name.split('.')[0]}.npy", anomaly_map)

                if save_examples and idx < 3:
                    num_layers = len(dists_per_layer)
                    cols = 3
                    rows = int(np.ceil((3 + num_layers) / cols))

                    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
                    axes = axes.flatten()

                    ###################################################################################
                    # Visualization & Diagnostic Plotting
                    ###################################################################################
                    
                    # 1. Original Input Image
                    axes[0].imshow(img_rgb)
                    axes[0].set_title("Test Image")
                    axes[0].axis("off")

                    # 2. Final Fused Anomaly Map (Weighted Average)
                    im = axes[1].imshow(anomaly_map_norm, cmap='jet')
                    axes[1].set_title("Avg Anomaly Map")
                    axes[1].axis("off")
                    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, orientation="horizontal")

                    # 3. Anomaly Score Distribution (Histogram)
                    axes[2].hist(anomaly_map.flatten(), bins=50)
                    axes[2].axvline(score, color='red', linestyle='dashed')
                    axes[2].set_title("Score Histogram")

                    # 4. Individual Layer Heatmaps
                    for i, d_masked in enumerate(dists_per_layer):
                        ax = axes[3 + i]
                        im = ax.imshow(d_masked, cmap='jet')
                        ax.set_title(f"Layer {i} Map")
                        ax.axis("off")
                        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, orientation="horizontal")

                    # 5. Clean up unused axes
                    for j in range(3 + num_layers, len(axes)):
                        axes[j].axis("off")
                    ###################################################################################

                    plt.tight_layout()
                    example_dir = f"{plots_dir}/{object_name}/examples"
                    os.makedirs(example_dir, exist_ok=True)
                    plt.savefig(f"{example_dir}/example_{anomaly_type}_{idx}.png")
                    plt.close()


    return anomaly_scores, time_memorybank, inference_times