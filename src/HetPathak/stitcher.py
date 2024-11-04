import numpy as np
import cv2
import glob
import os

class PanaromaStitcher:
    def __init__(self, fov=60, resize=800, ratio=0.72):
        self.fov = fov
        self.resize = resize
        self.ratio = ratio
        self.homography_matrices = []

    def cylindrical_projection(self, img):
        h, w = img.shape[:2]
        center_x, center_y = w // 2, h // 2
        focal_len = w / (2 * np.tan(np.radians(self.fov) / 2))
        img_proj = np.zeros_like(img)
        
        for y in range(h):
            for x in range(w):
                theta = (x - center_x) / focal_len
                h_proj = (y - center_y) / focal_len
                X = int(focal_len * np.tan(theta) + center_x)
                Y = int(focal_len * h_proj / np.cos(theta) + center_y)
                if 0 <= X < w and 0 <= Y < h:
                    img_proj[y, x] = img[Y, X]
                    
        return img_proj

    def sift_matches(self, img1, img2):
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
        kp2, des2 = sift.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)
        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(des1, des2, k=2)

        good_matches = [m for m, n in matches if m.distance < self.ratio * n.distance]
        return good_matches, kp1, kp2

    def normalize_coordinates(self, coords):
        """Normalize points for stable homography computation."""
        mean_val = np.mean(coords, axis=0)
        std_dev = np.std(coords)
        scale_factor = np.sqrt(2) / std_dev

        transform_matrix = np.array([
            [scale_factor, 0, -scale_factor * mean_val[0]],
            [0, scale_factor, -scale_factor * mean_val[1]],
            [0, 0, 1]
        ])
        
        normalized_coords = np.dot(transform_matrix, np.vstack((coords.T, np.ones((1, coords.shape[0])))))
        return transform_matrix, normalized_coords.T

    def estimate_homography_matrix(self, src_pts, dst_pts):
        """Compute homography matrix using DLT."""
        A_matrix = []
        for (x_src, y_src), (x_dst, y_dst) in zip(src_pts, dst_pts):
            A_matrix.append([-x_src, -y_src, -1, 0, 0, 0, x_dst * x_src, x_dst * y_src, x_dst])
            A_matrix.append([0, 0, 0, -x_src, -y_src, -1, y_dst * x_src, y_dst * y_src, y_dst])
        A_matrix = np.array(A_matrix)
        
        _, _, Vt = np.linalg.svd(A_matrix)
        homography = Vt[-1].reshape((3, 3))
        return homography / homography[2, 2]

    def ransac_homography(self, src_pts, dst_pts, threshold=4.0, max_iter=1000):
        """RANSAC for robust homography estimation."""
        best_inliers = []
        best_homography = None
        
        for _ in range(max_iter):
            sample_idx = np.random.choice(len(src_pts), 4, replace=False)
            sample_src = src_pts[sample_idx]
            sample_dst = dst_pts[sample_idx]
            
            homography_est = self.estimate_homography_matrix(sample_src, sample_dst)
            inliers = []
            
            for i, (src, dst) in enumerate(zip(src_pts, dst_pts)):
                proj = np.dot(homography_est, np.array([src[0], src[1], 1]))
                proj /= proj[2]
                error = np.linalg.norm(np.array([dst[0], dst[1]]) - proj[:2])
                if error < threshold:
                    inliers.append(i)
            
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_homography = homography_est

        if best_homography is not None:
            best_homography = self.estimate_homography_matrix(src_pts[best_inliers], dst_pts[best_inliers])
        
        return best_homography

    def find_homography(self, matches, kp1, kp2):
        """Find homography using custom RANSAC."""
        if len(matches) < 4:
            print("\nNot enough matches found between the images.\n")
            return None
            
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        h_matrix = self.ransac_homography(src_pts, dst_pts)
        return h_matrix

    def warp_and_stitch(self, base_img, img_to_stitch, h_matrix):
        h1, w1 = base_img.shape[:2]
        h2, w2 = img_to_stitch.shape[:2]

        corners = np.array([[0, 0, 1], [w2, 0, 1], [0, h2, 1], [w2, h2, 1]]).T
        warped_corners = np.dot(h_matrix, corners)
        warped_corners /= warped_corners[2]

        min_x = min(0, np.min(warped_corners[0]))
        min_y = min(0, np.min(warped_corners[1]))
        max_x = max(w1, np.max(warped_corners[0]))
        max_y = max(h1, np.max(warped_corners[1]))

        translation = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float32)
        output_size = (int(max_x - min_x), int(max_y - min_y))

        base_warped = cv2.warpPerspective(base_img, translation, output_size)
        img_to_stitch_warped = cv2.warpPerspective(img_to_stitch, translation @ h_matrix, output_size)

        mask_base = (base_warped > 0).astype(np.uint8)
        mask_stitch = (img_to_stitch_warped > 0).astype(np.uint8)

        overlap_area = mask_base & mask_stitch
        base_warped[overlap_area == 1] = (base_warped[overlap_area == 1] * 0.5 + img_to_stitch_warped[overlap_area == 1] * 0.5).astype(np.uint8)
        base_warped[mask_stitch == 1] = img_to_stitch_warped[mask_stitch == 1]

        return base_warped

    def make_panaroma_for_images_in(self, path):
        images = self.load_and_resize_images(path)
        mid_idx = len(images) // 2
        base_img = self.cylindrical_projection(images[mid_idx])

        for i in range(mid_idx - 1, -1, -1):
            matches, kp1, kp2 = self.sift_matches(base_img, images[i])
            if len(matches) >= 4:
                h_matrix = self.find_homography(matches, kp1, kp2)
                base_img = self.warp_and_stitch(images[i], base_img, h_matrix)
                self.homography_matrices.insert(0, h_matrix)

        for i in range(mid_idx + 1, len(images)):
            matches, kp1, kp2 = self.sift_matches(base_img, images[i])
            if len(matches) >= 4:
                h_matrix = self.find_homography(matches, kp1, kp2)
                base_img = self.warp_and_stitch(base_img, images[i], h_matrix)
                self.homography_matrices.append(h_matrix)

        return base_img, self.homography_matrices

    def load_and_resize_images(self, path):
        all_images = sorted(glob.glob(path + os.sep + '*'))
        return [cv2.resize(cv2.imread(img), (self.resize, self.resize)) for img in all_images]
