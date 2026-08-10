import os
import cv2
import numpy as np
import fitz # PyMuPDF for PDF to PNG conversion
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import cm
from reportlab.graphics import renderPDF
from svglib.svglib import svg2rlg

# --- CONFIGURATION ---
BASE_UNIT_CM = 1.7
KEY_SPACING_CM = 0.0 # Tiny gap between keycaps (adjust as needed)
CORNER_RADIUS_CM = 0.2
PAGE_WIDTH, PAGE_HEIGHT = landscape(A4)
ARUCO_SIZE_CM = 1.7
ARUCO_PADDING_CM = 0.05
LOGO_SVG_PATH = "Milky Uei Logo Black.svg" # Put your SVG path here
OUTPUT_PDF = "Keyboard_Layout_Precision.pdf"
OUTPUT_PNG = "Keyboard_Layout_Precision.png"

# Keyboard Layout based on your C++ code
# Use \n to split secondary/primary symbols for vertical stacking
KEYBOARD_LAYOUT = [
    [("¬\n`   ¦", 1.0), ("!\n1", 1.0), ('"\n2', 1.0), ("£\n3", 1.0), ("$\n4", 1.0), ("%\n5", 1.0), ("^\n6", 1.0), ("&\n7", 1.0), ("*\n8", 1.0), ("(\n9", 1.0), (")\n0", 1.0), ("_\n-", 1.0), ("+\n=", 1.0), ("backspace", 2.0)],
    [("tab", 1.5), ("Q", 1.0), ("W", 1.0), ("E", 1.0), ("R", 1.0), ("T", 1.0), ("Y", 1.0), ("U", 1.0), ("I", 1.0), ("O", 1.0), ("P", 1.0), ("{\n[", 1.0), ("}\n]", 1.0), ("enter", 1.5)],
    [("caps", 1.75), ("A", 1.0), ("S", 1.0), ("D", 1.0), ("F", 1.0), ("G", 1.0), ("H", 1.0), ("J", 1.0), ("K", 1.0), ("L", 1.0), (":\n;", 1.0), ("@\n'", 1.0), ("~\n#", 1.0), ("enter_bottom", 1.25)],
    [("shift", 1.25), ("|\n\\", 1.0), ("Z", 1.0), ("X", 1.0), ("C", 1.0), ("V", 1.0), ("B", 1.0), ("N", 1.0), ("M", 1.0), ("<\n,", 1.0), (">\n.", 1.0), ("?\n/", 1.0), ("shift", 2.75)],
    [("ctrl", 1.25), ("win", 1.25), ("alt", 1.25), ("Space", 6.25), ("alt gr", 1.25), ("fn", 1.25), ("option", 1.25), ("ctrl", 1.25)]
]

def generate_aruco_markers(num_markers, output_dir="markers"):
    """Generates ArUco markers and saves them as images."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_files = []
    
    for i in range(num_markers):
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, i, 200)
        
        border_size = 20
        marker_img_bordered = cv2.copyMakeBorder(
            marker_img, border_size, border_size, border_size, border_size, 
            cv2.BORDER_CONSTANT, value=[255, 255, 255]
        )
        
        filepath = os.path.join(output_dir, f"marker_{i}.png")
        cv2.imwrite(filepath, marker_img_bordered)
        marker_files.append(filepath)
        
    return marker_files

def draw_keyboard(c):
    total_width_u = sum([key[1] for key in KEYBOARD_LAYOUT[0]]) 
    total_height_u = len(KEYBOARD_LAYOUT) 
    
    board_width_cm = total_width_u * BASE_UNIT_CM
    board_height_cm = total_height_u * BASE_UNIT_CM
    
    start_x = (PAGE_WIDTH - (board_width_cm * cm)) / 2
    start_y = (PAGE_HEIGHT + (board_height_cm * cm)) / 2 
    
    c.setLineWidth(1.5)
    
    current_y = start_y
    for row in KEYBOARD_LAYOUT:
        current_x = start_x
        current_y -= BASE_UNIT_CM * cm
        
        for key_label, width_u in row:
            key_width = width_u * BASE_UNIT_CM * cm
            key_height = BASE_UNIT_CM * cm
            
            if key_label == "enter_bottom":
                # The shape was already drawn by the main 'enter' key. 
                # Just advance the X coordinate to maintain the grid.
                current_x += key_width
                continue
                
            elif key_label == "enter":
                # --- SPECIAL CASE: Flawless Continuous Vector ISO L-Shape ---
                s = KEY_SPACING_CM * cm
                r = CORNER_RADIUS_CM * cm
                
                # Calculate exact boundaries of the key
                x_lt = current_x + s
                x_r = current_x + (1.5 * BASE_UNIT_CM * cm) - s
                x_lb = current_x + (0.25 * BASE_UNIT_CM * cm) + s
                
                y_t = current_y + key_height - s
                y_b = current_y - key_height + s
                y_m = current_y - s
                
                # Draw a single continuous path
                p = c.beginPath()
                p.moveTo(x_lt + r, y_t) # Start top-left
                p.lineTo(x_r - r, y_t) # Top edge
                p.arcTo(x_r - 2*r, y_t - 2*r, x_r, y_t, 90, -90) # Top-Right corner
                p.lineTo(x_r, y_b + r) # Right edge
                p.arcTo(x_r - 2*r, y_b, x_r, y_b + 2*r, 0, -90) # Bottom-Right corner
                p.lineTo(x_lb + r, y_b) # Bottom edge
                p.arcTo(x_lb, y_b, x_lb + 2*r, y_b + 2*r, 270, -90) # Bottom-Left corner (lower half)
                p.lineTo(x_lb, y_m) # Left edge of lower half
                p.lineTo(x_lt + r, y_m) # Inner horizontal corner
                p.arcTo(x_lt, y_m, x_lt + 2*r, y_m + 2*r, 270, -90) # Bottom-Left corner (upper half)
                p.lineTo(x_lt, y_t - r) # Left edge of upper half
                p.arcTo(x_lt, y_t - 2*r, x_lt + 2*r, y_t, 180, -90) # Top-Left corner
                p.close()
                
                c.drawPath(p, stroke=1, fill=0)
                
                # Draw Text
                c.setFillColorRGB(0, 0, 0) # Black text
                c.setFont("Helvetica-Bold", 10)
                text_str = "enter"
                text_width = c.stringWidth(text_str, "Helvetica-Bold", 10)
                
                text_x = current_x + (key_width - text_width) / 2.0
                text_y = current_y + (key_height * 0.25) # 25% up from the inner bend
                c.drawString(text_x, text_y, text_str)
                
            else:
                # --- STANDARD KEYS ---
                c.roundRect(
                    current_x + (KEY_SPACING_CM * cm), 
                    current_y + (KEY_SPACING_CM * cm), 
                    key_width - (2 * KEY_SPACING_CM * cm), 
                    key_height - (2 * KEY_SPACING_CM * cm), 
                    CORNER_RADIUS_CM * cm, 
                    stroke=1, fill=0
                )
                
                if key_label == "Space" and os.path.exists(LOGO_SVG_PATH):
                    drawing = svg2rlg(LOGO_SVG_PATH)
                    if drawing:
                        logo_scale = (key_height * 0.8) / drawing.height 
                        drawing.scale(logo_scale, logo_scale)
                        logo_x = current_x + (key_width / 2) - ((drawing.width * logo_scale) / 2)
                        logo_y = current_y + (key_height / 2) - ((drawing.height * logo_scale) / 2)
                        renderPDF.draw(drawing, c, logo_x, logo_y)
                        
                elif key_label == "¬\n`   ¦":
                    c.setFont("Helvetica-Bold", 10)
                    padding_x = 0.4 * cm 
                    padding_y = 0.3 * cm 
                    
                    # Top Left: ¬
                    c.drawString(current_x + padding_x, current_y + key_height - padding_y - 8, "¬")
                    # Bottom Left: `
                    c.drawString(current_x + padding_x, current_y + padding_y, "`")
                    # Bottom Right: ¦
                    broken_bar_width = c.stringWidth("¦", "Helvetica-Bold", 10)
                    c.drawString(current_x + key_width - padding_x - broken_bar_width, current_y + padding_y, "¦")
                    
                else:
                    c.setFont("Helvetica-Bold", 10)
                    
                    if "\n" in key_label:
                        lines = key_label.split("\n")
                        num_lines = len(lines)
                        line_height = 16 
                        
                        total_text_height = num_lines * line_height
                        start_text_y = current_y + (key_height / 2.0) + (total_text_height / 2.0) - (line_height * 0.8)
                        
                        for idx, line in enumerate(lines):
                            text_width = c.stringWidth(line, "Helvetica-Bold", 10)
                            text_x = current_x + (key_width - text_width) / 2.0
                            text_y = start_text_y - (idx * line_height)
                            c.drawString(text_x, text_y, line)
                    else:
                        text_width = c.stringWidth(key_label, "Helvetica-Bold", 10)
                        c.drawString(
                            current_x + (key_width - text_width) / 2.0, 
                            current_y + (key_height / 2.0) - 3, 
                            key_label
                        )
            
            current_x += key_width

    return start_x, start_y - (total_height_u * BASE_UNIT_CM * cm), board_width_cm * cm, board_height_cm * cm

def draw_aruco_markers(c, marker_files, board_x, board_y, board_w, board_h):
    marker_size = ARUCO_SIZE_CM * cm
    padding = ARUCO_PADDING_CM * cm
    
    left_x = board_x - padding - marker_size
    right_x = board_x + board_w + padding
    bottom_y = board_y - padding - marker_size
    top_y = board_y + board_h + padding
    
    step_x = (right_x - left_x) / 8.0 
    step_y = (top_y - bottom_y) / 3.0
    
    positions = []
    
    # 1. First row (Top row): IDs 0 to 8 (Left to Right)
    for i in range(9):
        positions.append((left_x + (i * step_x), top_y))
        
    # 2. Second row (Upper middle): IDs 9 to 10 (Left, then Right)
    positions.append((left_x, top_y - step_y))        # ID 9
    positions.append((right_x, top_y - step_y))       # ID 10
        
    # 3. Third row (Lower middle): IDs 11 to 12 (Left, then Right)
    positions.append((left_x, top_y - (2 * step_y)))  # ID 11
    positions.append((right_x, top_y - (2 * step_y))) # ID 12
        
    # 4. Fourth row (Bottom row): IDs 13 to 21 (Left to Right)
    for i in range(9):
        positions.append((left_x + (i * step_x), bottom_y))

    # Draw the 22 collected positions in the new specific order
    for idx, pos in enumerate(positions):
        if idx < len(marker_files):
            c.drawImage(marker_files[idx], pos[0], pos[1], width=marker_size, height=marker_size)

def main():
    print("Generating ArUco markers...")
    marker_files = generate_aruco_markers(22)
    
    print(f"Creating PDF layout at {OUTPUT_PDF}...")
    c = canvas.Canvas(OUTPUT_PDF, pagesize=landscape(A4))
    
    board_x, board_y, board_w, board_h = draw_keyboard(c)
    draw_aruco_markers(c, marker_files, board_x, board_y, board_w, board_h)
    
    c.save()
    print("Done! PDF saved successfully.")
    
    print(f"Converting PDF to PNG at {OUTPUT_PNG}...")
    doc = fitz.open(OUTPUT_PDF)
    page = doc.load_page(0) # Load the first page
    pix = page.get_pixmap(dpi=300) # 300 DPI for high print quality
    pix.save(OUTPUT_PNG)
    doc.close()
    print("Done! PNG saved successfully.")

if __name__ == "__main__":
    main()