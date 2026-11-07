"""
Deepfake Guardian AI - Detection Core: DBSCAN Anomaly Analysis
Implements Scikit-Learn DBSCAN clustering to detect unusual noise, compression,
and frequency patterns as specified in the architecture blueprint:
[DBSCAN (Anomaly Analysis) -> Detect Unusual Patterns]
"""

import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import io
from typing import Dict, Any, Tuple

import config

class DBSCANAnomalyAnalyzer:
    """
    Performs spatial and frequency feature clustering using DBSCAN.
    Identifies unnatural noise boundaries, GAN synthesis artifacts, and outlier patches.
    """

    def __init__(self, eps: float = config.DBSCAN_EPS, min_samples: int = config.DBSCAN_MIN_SAMPLES):
        self.eps = eps
        self.min_samples = min_samples

    def extract_patch_features(self, image_rgb: np.ndarray, grid_size: int = 12) -> Tuple[np.ndarray, list]:
        """
        Divides the face crop into a grid of patches and extracts localized forensic features:
        - Mean RGB color
        - Local standard deviation (texture roughness)
        - High-frequency Laplacian variance (edge/sensor noise)
        - Local gradient energy
        """
        h, w = image_rgb.shape[:2]
        step_y = h // grid_size
        step_x = w // grid_size

        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        feature_vectors = []
        patch_coords = []

        for i in range(grid_size):
            for j in range(grid_size):
                y1 = i * step_y
                y2 = min((i + 1) * step_y, h)
                x1 = j * step_x
                x2 = min((j + 1) * step_x, w)

                patch_rgb = image_rgb[y1:y2, x1:x2]
                patch_lap = laplacian[y1:y2, x1:x2]

                if patch_rgb.size == 0:
                    continue

                mean_r = np.mean(patch_rgb[:, :, 0])
                mean_g = np.mean(patch_rgb[:, :, 1])
                mean_b = np.mean(patch_rgb[:, :, 2])
                std_rgb = np.std(patch_rgb)
                lap_var = np.var(patch_lap)
                edge_energy = np.mean(np.abs(patch_lap))

                # Feature vector (normalized texture and residual fingerprint)
                vec = [mean_r, mean_g, mean_b, std_rgb, lap_var, edge_energy]
                feature_vectors.append(vec)
                patch_coords.append((x1, y1, x2, y2))

        return np.array(feature_vectors), patch_coords

    def analyze(self, image_rgb: np.ndarray) -> Dict[str, Any]:
        """
        Runs DBSCAN clustering on patch feature vectors and evaluates anomaly indicators.
        """
        features, patch_coords = self.extract_patch_features(image_rgb)
        
        if len(features) < 10:
            return {
                "clusters_count": 0,
                "noise_count": 0,
                "anomaly_percentage": 0.0,
                "severity": "NORMAL",
                "plot_buf": None
            }

        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        # DBSCAN Clustering
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = db.fit_predict(scaled_features)

        # Labels: -1 indicates noise / anomaly outlier
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int(np.sum(labels == -1))
        total_points = len(labels)
        anomaly_pct = float((n_noise / total_points) * 100.0)

        # Categorize anomaly severity
        if anomaly_pct > 25.0:
            severity = "CRITICAL (High Deepfake Artifact Density)"
        elif anomaly_pct > 14.0:
            severity = "ELEVATED (Unusual Structural Inconsistencies)"
        elif anomaly_pct > 6.0:
            severity = "MODERATE (Localized Boundary Variance)"
        else:
            severity = "LOW (Natural Organic Uniformity)"

        # 2D Projection for visual dashboard plot
        pca = PCA(n_components=2)
        projected_2d = pca.fit_transform(scaled_features)

        # Generate cluster visualization plot matching the architecture diagram
        plot_buf = self._generate_cluster_plot(projected_2d, labels, n_clusters, n_noise)

        return {
            "clusters_count": int(n_clusters),
            "noise_count": int(n_noise),
            "total_patches": int(total_points),
            "anomaly_percentage": round(anomaly_pct, 2),
            "severity": severity,
            "plot_bytes": plot_buf,
            "labels": labels.tolist(),
            "projected_points": projected_2d.tolist()
        }

    def _generate_cluster_plot(
        self, 
        points_2d: np.ndarray, 
        labels: np.ndarray, 
        n_clusters: int, 
        n_noise: int
    ) -> bytes:
        """
        Renders a dark cybersecurity scatter plot of the DBSCAN clusters and anomalous noise points.
        """
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(6, 4.2), dpi=120)
        fig.patch.set_facecolor('#0b0f19')
        ax.set_facecolor('#0f172a')

        unique_labels = set(labels)
        try:
            colors = plt.colormaps['tab10']
        except Exception:
            colors = plt.cm.tab10

        for k in unique_labels:
            class_member_mask = (labels == k)
            xy = points_2d[class_member_mask]

            if k == -1:
                # Black/Red 'x' for anomalous outlier noise points
                ax.scatter(
                    xy[:, 0], xy[:, 1], 
                    c='#ef4444', 
                    marker='x', 
                    s=55, 
                    linewidths=2, 
                    label=f'Anomalous Noise (n={n_noise})'
                )
            else:
                col = colors(k % 10)
                ax.scatter(
                    xy[:, 0], xy[:, 1], 
                    color=col, 
                    alpha=0.85, 
                    s=35, 
                    edgecolors='none', 
                    label=f'Cluster {k+1}'
                )

        ax.set_title("DBSCAN Anomaly Distribution Map", fontsize=11, color='#60a5fa', weight='bold', pad=12)
        ax.set_xlabel("Principal Component 1 (Spatial Texture Variance)", fontsize=8, color='#94a3b8')
        ax.set_ylabel("Principal Component 2 (Residual Noise Gradient)", fontsize=8, color='#94a3b8')
        ax.tick_params(colors='#64748b', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#1e293b')

        ax.grid(True, linestyle='--', alpha=0.2, color='#475569')
        ax.legend(loc='upper right', fontsize=7, facecolor='#1e293b', edgecolor='#334155')

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
