import cv2
import numpy as np

def nothing(x):
    pass

def confirmation_window():
    cv2.namedWindow("Xac nhan chay chuong trinh")
    time_input = ""
    
    while True:
        # backgorund
        confirm_image = np.ones((200, 480, 3), dtype=np.uint8) * 255
        cv2.putText(confirm_image, 
                    "Nhan 'O' de bat dau, 'X' de thoat", 
                    (20, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(confirm_image, 
                    "Time muon bat dau video(s): " + time_input, 
                    (20, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        cv2.imshow("Xac nhan chay chuong trinh", confirm_image)
        
        key = cv2.waitKey(100)
        
        # check presskey
        if key >= ord('0') and key <= ord('9'):
            time_input += chr(key)
        elif key == ord('o'):  
            if time_input.isdigit():
                start_time = int(time_input)
                cv2.destroyWindow("Xac nhan chay chuong trinh")
                return start_time
            else:
                start_time = 0
                cv2.destroyWindow("Xac nhan chay chuong trinh")
                return start_time
        elif key == ord('o') or key == ord('O'):
            cv2.destroyWindow("Xac nhan chay chuong trinh")
            return 0  
        elif key == ord('x') or key == ord('X'):
            cv2.destroyWindow("Xac nhan chay chuong trinh")
            return None

start_time = confirmation_window()
if start_time is None:
    print("exits")
    exit()

cap = cv2.VideoCapture("XLA/XLA.mp4")
#pic in frame
overlay = cv2.imread('KTXLA/sensor.png', cv2.IMREAD_UNCHANGED) 
overlay_resized = cv2.resize(overlay, (75, 75)) 
if overlay_resized.shape[2] == 4:  
    overlay_resized = cv2.cvtColor(overlay_resized, cv2.COLOR_RGBA2RGB)

overlay1 = cv2.imread('KTXLA/sensor1.png', cv2.IMREAD_UNCHANGED) 
overlay1_resized = cv2.resize(overlay1, (75, 75)) 
if overlay1_resized.shape[2] == 4:  
    overlay1_resized = cv2.cvtColor(overlay1_resized, cv2.COLOR_RGBA2RGB)

overlay2 = cv2.imread('KTXLA/pic.jpg', cv2.IMREAD_UNCHANGED) 
overlay2_resized = cv2.resize(overlay2, (720, 372)) 
if overlay2_resized.shape[2] == 4:  
    overlay2_resized = cv2.cvtColor(overlay2_resized, cv2.COLOR_RGBA2RGB)

#time
if start_time > 0:
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

cv2.namedWindow("Thanh dieu chinh mat na")
cv2.createTrackbar("H Lower", "Thanh dieu chinh mat na", 69, 179, nothing)
cv2.createTrackbar("H Upper", "Thanh dieu chinh mat na", 179, 179, nothing)
cv2.createTrackbar("S Lower", "Thanh dieu chinh mat na", 39, 255, nothing)
cv2.createTrackbar("S Upper", "Thanh dieu chinh mat na", 255, 255, nothing)
cv2.createTrackbar("V Lower", "Thanh dieu chinh mat na", 0, 255, nothing)
cv2.createTrackbar("V Upper", "Thanh dieu chinh mat na", 255, 255, nothing)

# trackbar fps
cv2.createTrackbar("FPS", "Thanh dieu chinh mat na", 30, 240, nothing)

shape_counts = {
    "HCN": 0,
    "Hinh Tron": 0
}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (720, 372))
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h_lower = cv2.getTrackbarPos("H Lower", "Thanh dieu chinh mat na")
    h_upper = cv2.getTrackbarPos("H Upper", "Thanh dieu chinh mat na")
    s_lower = cv2.getTrackbarPos("S Lower", "Thanh dieu chinh mat na")
    s_upper = cv2.getTrackbarPos("S Upper", "Thanh dieu chinh mat na")
    v_lower = cv2.getTrackbarPos("V Lower", "Thanh dieu chinh mat na")
    v_upper = cv2.getTrackbarPos("V Upper", "Thanh dieu chinh mat na")

    lower_bound = np.array([h_lower, s_lower, v_lower])
    upper_bound = np.array([h_upper, s_upper, v_upper])

    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
    mask_inv = cv2.bitwise_not(mask)
    mask_inv = cv2.medianBlur(mask_inv, 5)

    kernel = np.ones((2, 2), np.uint8)
    mask_inv = cv2.morphologyEx(mask_inv, cv2.MORPH_OPEN, kernel)
    mask_inv = cv2.morphologyEx(mask_inv, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_image = frame.copy()
    shapes_image = np.zeros_like(frame)
    centroids = []
    
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 96:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])  
                    cy = int(M["m01"] / M["m00"])  
                    centroids.append((cx, cy))
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(contour_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(contour_image, f'({cx}, {cy})', (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                    shape = "khong xac dinh"
                    
                    if len(approx) == 3:
                        shape = "Tam Giac"
                        cv2.drawContours(shapes_image, [contour], -1, (0, 255, 0), 2) 
                    elif len(approx) <= 6:
                        x, y, w, h = cv2.boundingRect(approx)
                        aspect_ratio = float(w) / h
                        shape = "Vuong" if aspect_ratio >= 0.95 and aspect_ratio <= 1.05 else "HCN"
                        cv2.drawContours(shapes_image, [contour], -1, (255, 0, 0), 2)  
                    elif len(approx) >= 7:
                        shape = "Hinh Tron"
                        cv2.drawContours(shapes_image, [contour], -1, (0, 0, 255), 2)

                    # training vitri
                    if (415<=cx<=417 and cy==183)or(415<=cx<=416 and cy==158)or(420<=cx<=424 and cy==207)or(415<=cx<=424 and cy==186)\
                         or(423<=cx<=424 and cy==228)or(420<=cx<=425 and cy==193)or(420<=cx<=423 and cy==167)or(429<=cx<=430 and cy==220)\
                            or(418<=cx<=421 and cy==196)or(415<=cx<=416 and cy==229)or(428<=cx<=430 and cy==174)or(414<=cx<=415 and cy==211)\
                                or(426<=cx<=428 and (cy==173 or cy==245))or(421<=cx<=425 and cy==210)or(425<=cx<=430 and cy==181)or(425<=cx<=428 and cy==219)or(428<=cx<=429 and cy==256):
                        if shape in shape_counts:
                            shape_counts[shape] += 1

                    cv2.putText(shapes_image, shape, (cx - 20, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.line(contour_image, (382, 0), (382, frame.shape[0]), (255, 255, 0), 5)
    cv2.line(contour_image, (402, 0), (402, frame.shape[0]), (255, 255, 0), 5)
    cv2.line(contour_image, (422, 0), (422, frame.shape[0]), (255, 255, 0), 5)
    cv2.line(contour_image, (442, 0), (442, frame.shape[0]), (255, 255, 0), 5)
    # tao newframe
    overlay_height, overlay_width = overlay_resized.shape[:2]
    combined_canvas = np.zeros((744, 1440, 3), dtype=np.uint8)
    # resize
    mask_resized = cv2.cvtColor(cv2.resize(mask_inv, (720, 372)), cv2.COLOR_GRAY2BGR)
    contour_resized = cv2.resize(contour_image, (720, 372))
    shapes_resized = cv2.resize(shapes_image, (720, 372))

    # place newframe
    combined_canvas[0:372, 0:720] = mask_resized  
    combined_canvas[0:372, 720:1440] = contour_resized  
    combined_canvas[372:744, 0:720] = shapes_resized 
    #sensor
    x_offset = 1095
    y_offset = 297
    if (y_offset + overlay_resized.shape[0] <= combined_canvas.shape[0]) and (x_offset + overlay_resized.shape[1] <= combined_canvas.shape[1]):
        combined_canvas[y_offset:y_offset + overlay_resized.shape[0], x_offset:x_offset + overlay_resized.shape[1]] = overlay_resized
    else:
        print("x")
    #sensor1
    x1_offset = 1095
    y1_offset = 0
    if (y_offset + overlay1_resized.shape[0] <= combined_canvas.shape[0]) and (x1_offset + overlay1_resized.shape[1] <= combined_canvas.shape[1]):
        combined_canvas[y1_offset:y1_offset + overlay1_resized.shape[0], x1_offset:x1_offset + overlay1_resized.shape[1]] = overlay1_resized
    else:
        print("x")
    #pic
    x2_offset = 720
    y2_offset = 372
    if (y_offset + overlay2_resized.shape[0] <= combined_canvas.shape[0]) and (x2_offset + overlay2_resized.shape[1] <= combined_canvas.shape[1]):
        combined_canvas[y2_offset:y2_offset + overlay2_resized.shape[0], x2_offset:x2_offset + overlay2_resized.shape[1]] = overlay2_resized
    else:
        print("x")

    # display shape counts
    cv2.putText(combined_canvas, f'Phan loai hinh dang:', (730, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,215,255), 2)
    y_offset = 70
    for shape, count in shape_counts.items():
        cv2.putText(combined_canvas, f'{shape}: {count}', (730, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,215,255), 2)
        y_offset += 30

    cv2.putText(combined_canvas, f'Mat Na Loc Nhieu (Mask)', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    image = cv2.rectangle(combined_canvas, (1,1), (715,368), (255, 255, 255), 4)
    image = cv2.rectangle(combined_canvas, (1,372), (715,740), (128,128,240), 4)
    image = cv2.rectangle(combined_canvas, (720,1), (1435,367), (47,255,173), 4)
    cv2.putText(combined_canvas, f'Khoanh vung nhan dang', (50, 422), cv2.FONT_HERSHEY_SIMPLEX, 1, (226,43,138), 2)

    total_count = sum(shape_counts.values())
    print(total_count)
    for soluong in shape_counts.values():
        cv2.putText(combined_canvas, f'So luong vat the da di qua cam bien: {total_count}', (50, 472), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (226,43,138), 2)
        y_offset += 30
    
    cv2.putText(combined_canvas, f'Pham The Vinh, MSSV: 21161388', (780, 420), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

    cv2.imshow("Chuong trinh", combined_canvas)
    cv2.moveWindow("Thanh dieu chinh mat na", 1020, 450)
    
    fps = cv2.getTrackbarPos("FPS", "Thanh dieu chinh mat na")
    delay = int(240 / fps) 
    print("Toc do video moi giay: ", delay / 1)
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
