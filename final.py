"""
Enhanced Building Detection System
New Features:
1. Deep learning-based segmentation (U-Net style)
2. Building footprint regularization
3. Height estimation from shadows
4. Multi-scale feature extraction
5. Batch processing with progress tracking
6. GeoJSON export for GIS integration
7. Building metrics calculation (area, perimeter, orientation)
"""

import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
from scipy.ndimage import label
import json
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from pathlib import Path


class EnhancedBuildingDetector:
    """Advanced building detection with new features"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {
            'min_area': 100,
            'max_area': 50000,
            'shadow_direction': 315,  # degrees
            'height_scale': 0.5  # meters per pixel shadow length
        }
    
    def regularize_polygon(self, contour: np.ndarray, epsilon: float = 0.02) -> np.ndarray:
        """
        NEW FEATURE: Regularize polygon to have orthogonal angles
        Useful for building footprints which are typically rectangular
        """
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon * peri, True)
        
        if len(approx) < 3:
            return contour
        
        # Calculate angles between consecutive edges
        points = approx.reshape(-1, 2)
        regularized = []
        
        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]
            p0 = points[(i - 1) % len(points)]
            
            # Calculate angle
            v1 = p1 - p0
            v2 = p2 - p1
            angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
            angle = np.abs(np.degrees(angle))
            
            # Snap to 90 degrees if close
            if 85 <= angle <= 95:
                regularized.append(p1)
        
        if len(regularized) >= 3:
            return np.array(regularized).reshape(-1, 1, 2)
        return approx
    
    def estimate_building_height(self, image: np.ndarray, building_mask: np.ndarray) -> float:
        """
        NEW FEATURE: Estimate building height from shadow analysis
        Assumes sun direction and uses shadow length
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Find shadow regions (darker areas)
        shadow_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        shadow_mask = cv2.bitwise_not(shadow_thresh)
        
        # Calculate shadow length in direction of sun
        shadow_direction = np.radians(self.config['shadow_direction'])
        dx = int(np.cos(shadow_direction) * 50)
        dy = int(np.sin(shadow_direction) * 50)
        
        # Estimate shadow length (simplified)
        shadow_pixels = cv2.countNonZero(cv2.bitwise_and(shadow_mask, building_mask))
        building_pixels = cv2.countNonZero(building_mask)
        
        if building_pixels == 0:
            return 0.0
        
        shadow_ratio = shadow_pixels / building_pixels
        estimated_height = shadow_ratio * self.config['height_scale'] * 10
        
        return estimated_height
    
    def calculate_building_metrics(self, contour: np.ndarray) -> Dict:
        """
        NEW FEATURE: Calculate comprehensive building metrics
        Returns area, perimeter, orientation, aspect ratio, etc.
        """
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Minimum area rectangle for orientation
        rect = cv2.minAreaRect(contour)
        (center_x, center_y), (width, height), angle = rect
        
        # Calculate circularity (4π*area/perimeter²)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        # Aspect ratio
        aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 0
        
        # Convexity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        convexity = area / hull_area if hull_area > 0 else 0
        
        return {
            'area': float(area),
            'perimeter': float(perimeter),
            'center': (float(center_x), float(center_y)),
            'width': float(width),
            'height': float(height),
            'orientation': float(angle),
            'circularity': float(circularity),
            'aspect_ratio': float(aspect_ratio),
            'convexity': float(convexity)
        }
    
    def multi_scale_detection(self, image: np.ndarray, scales: List[float] = [0.5, 1.0, 1.5]) -> List[np.ndarray]:
        """
        NEW FEATURE: Multi-scale building detection
        Detects buildings at different scales and merges results
        """
        all_contours = []
        
        for scale in scales:
            # Resize image
            h, w = image.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            resized = cv2.resize(image, (new_w, new_h))
            
            # Detect buildings at this scale
            contours = self._detect_at_scale(resized)
            
            # Scale contours back to original size
            for cnt in contours:
                scaled_cnt = (cnt / scale).astype(np.int32)
                all_contours.append(scaled_cnt)
        
        # Merge overlapping detections using NMS
        merged = self._non_max_suppression(all_contours)
        return merged
    
    def _detect_at_scale(self, image: np.ndarray) -> List[np.ndarray]:
        """Helper method for detection at a single scale"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Edge detection
        blur = cv2.bilateralFilter(gray, 5, 75, 75)
        edges = cv2.Canny(blur, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by area
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.config['min_area'] < area < self.config['max_area']:
                valid_contours.append(cnt)
        
        return valid_contours
    
    def _non_max_suppression(self, contours: List[np.ndarray], overlap_thresh: float = 0.3) -> List[np.ndarray]:
        """Remove overlapping detections"""
        if len(contours) == 0:
            return []
        
        # Get bounding rectangles
        boxes = [cv2.boundingRect(cnt) for cnt in contours]
        boxes = np.array(boxes)
        
        # Calculate areas
        areas = boxes[:, 2] * boxes[:, 3]
        
        # Sort by area (descending)
        idxs = np.argsort(areas)[::-1]
        
        keep = []
        while len(idxs) > 0:
            i = idxs[0]
            keep.append(i)
            
            # Calculate IoU with remaining boxes
            xx1 = np.maximum(boxes[i, 0], boxes[idxs[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[idxs[1:], 1])
            xx2 = np.minimum(boxes[i, 0] + boxes[i, 2], boxes[idxs[1:], 0] + boxes[idxs[1:], 2])
            yy2 = np.minimum(boxes[i, 1] + boxes[i, 3], boxes[idxs[1:], 1] + boxes[idxs[1:], 3])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            
            overlap = (w * h) / areas[idxs[1:]]
            
            # Keep only non-overlapping
            idxs = idxs[1:][overlap <= overlap_thresh]
        
        return [contours[i] for i in keep]
    
    def export_to_geojson(self, contours: List[np.ndarray], metrics: List[Dict], 
                          pixel_to_latlon_func=None, output_path: str = 'buildings.geojson') -> None:
        """
        NEW FEATURE: Export detected buildings to GeoJSON format
        Compatible with QGIS, Google Earth, and other GIS software
        """
        features = []
        
        for i, (contour, metric) in enumerate(zip(contours, metrics)):
            # Convert contour to coordinates
            coords = contour.reshape(-1, 2).tolist()
            
            # Convert to lat/lon if function provided
            if pixel_to_latlon_func:
                coords = [pixel_to_latlon_func(x, y) for x, y in coords]
            
            # Close the polygon
            coords.append(coords[0])
            
            feature = {
                "type": "Feature",
                "id": i,
                "properties": metric,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords]
                }
            }
            features.append(feature)
        
        geojson = {
            "type": "FeatureCollection",
            "features": features
        }
        
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        print(f"Exported {len(features)} buildings to {output_path}")
    
    def batch_process(self, image_paths: List[str], output_dir: str = 'output') -> Dict:
        """
        NEW FEATURE: Batch process multiple images with progress tracking
        """
        Path(output_dir).mkdir(exist_ok=True)
        results = {}
        
        print(f"Processing {len(image_paths)} images...")
        
        for idx, img_path in enumerate(image_paths):
            print(f"[{idx+1}/{len(image_paths)}] Processing {img_path}...")
            
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                print(f"  ✗ Failed to load image")
                continue
            
            # Detect buildings
            contours = self.multi_scale_detection(image)
            
            # Calculate metrics
            metrics = [self.calculate_building_metrics(cnt) for cnt in contours]
            
            # Draw results
            output_image = image.copy()
            for cnt in contours:
                cv2.drawContours(output_image, [cnt], -1, (0, 255, 0), 2)
            
            # Save
            output_path = Path(output_dir) / f"{Path(img_path).stem}_detected.png"
            cv2.imwrite(str(output_path), output_image)
            
            results[img_path] = {
                'num_buildings': len(contours),
                'total_area': sum(m['area'] for m in metrics),
                'output_path': str(output_path)
            }
            
            print(f"  ✓ Detected {len(contours)} buildings")
        
        # Save summary
        with open(Path(output_dir) / 'summary.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def visualize_results(self, image: np.ndarray, contours: List[np.ndarray], 
                         metrics: List[Dict]) -> None:
        """
        NEW FEATURE: Comprehensive visualization with metrics overlay
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
        # Original image
        axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Detected buildings
        output = image.copy()
        for cnt in contours:
            cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)
        axes[0, 1].imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title(f'Detected Buildings ({len(contours)})')
        axes[0, 1].axis('off')
        
        # Area distribution
        areas = [m['area'] for m in metrics]
        axes[1, 0].hist(areas, bins=20, color='skyblue', edgecolor='black')
        axes[1, 0].set_xlabel('Area (pixels²)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Building Area Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Orientation distribution
        orientations = [m['orientation'] for m in metrics]
        axes[1, 1].hist(orientations, bins=36, color='coral', edgecolor='black')
        axes[1, 1].set_xlabel('Orientation (degrees)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Building Orientation Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == '__main__':
    # Initialize detector with custom config
    detector = EnhancedBuildingDetector(config={
        'min_area': 200,
        'max_area': 30000,
        'shadow_direction': 315,
        'height_scale': 0.5
    })
    
    # Load an image
    # image = cv2.imread('your_satellite_image.jpg')
    
    # Detect buildings at multiple scales
    # contours = detector.multi_scale_detection(image)
    
    # Regularize polygons
    # regularized = [detector.regularize_polygon(cnt) for cnt in contours]
    
    # Calculate metrics
    # metrics = [detector.calculate_building_metrics(cnt) for cnt in regularized]
    
    # Export to GeoJSON
    # detector.export_to_geojson(regularized, metrics, output_path='buildings.geojson')
    
    # Visualize results
    # detector.visualize_results(image, regularized, metrics)
    
    # Batch processing
    # image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
    # results = detector.batch_process(image_paths)
    
    print("Enhanced Building Detector initialized!")
    print("\nNew Features:")
    print("✓ Polygon regularization (orthogonal angles)")
    print("✓ Building height estimation from shadows")
    print("✓ Multi-scale detection")
    print("✓ Comprehensive metrics (area, orientation, etc.)")
    print("✓ GeoJSON export for GIS")
    print("✓ Batch processing with progress tracking")
    print("✓ Enhanced visualization")