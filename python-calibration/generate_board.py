import cv2
import numpy as np

def create_checkerboard(squares_x=10, squares_y=7, square_size_px=200):
    # Create an empty white image
    width = squares_x * square_size_px
    height = squares_y * square_size_px
    board = np.ones((height, width), dtype=np.uint8) * 255

    # Fill in the black squares
    for y in range(squares_y):
        for x in range(squares_x):
            if (x + y) % 2 == 0:
                cv2.rectangle(board,
                              (x * square_size_px, y * square_size_px),
                              ((x + 1) * square_size_px, (y + 1) * square_size_px),
                              0, -1)
    
    # Add a white border so the edge squares aren't cut off by the printer
    board_with_border = cv2.copyMakeBorder(board, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=255)
    
    filename = "checkerboard_to_print.png"
    cv2.imwrite(filename, board_with_border)
    print(f"Saved '{filename}'.")
    print("IMPORTANT: When printing, ensure 'Scale to Fit' is turned OFF (print at 100% scale).")

if __name__ == "__main__":
    create_checkerboard()