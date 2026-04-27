import cv2
import numpy as np

# 改成你的实际图片完整路径
img = cv2.imread(r'D:\Da_4\lunwen\K_laoshi\test\tu4\frames.123.png', cv2.IMREAD_GRAYSCALE)

img = cv2.resize(img, (188, 120))
blur = cv2.GaussianBlur(img, (5,5), 1.2)
edge_gt = cv2.Canny(blur, 20, 60)

combined = np.hstack([img, edge_gt])
cv2.imwrite(r'D:\Da_4\lunwen\K_laoshi\test\tu4\fig4-1_test_image.png', combined)
print('保存成功')