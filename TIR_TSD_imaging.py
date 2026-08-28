import matplotlib.pyplot as plt
import matplotlib.patches as patches
import re
import os

# ==========================================
# Configuration / Settings
# ==========================================

# File settings
TARGET_FILE = "./Strict_Evidence_Summary_human_only.txt"

# Scaling parameters for visualization
COMPRESS_RATE_EPVE = 1000    # Compression rate for EPVE region
COMPRESS_RATE_GAP  = 100     # Compression rate for gap region (>= 50 bp)

# Color configuration
COLOR_TSD        = 'mediumseagreen' # TSD block fill color
COLOR_TIR        = 'dodgerblue'     # TIR block fill color
COLOR_EPVE       = 'darkorange'     # EPVE block fill color
COLOR_LINE       = 'black'          # Gap line color
COLOR_EDGE       = 'black'          # Block border color
COLOR_TEXT_WHITE = 'white'          # Text color inside blocks

# Font size configuration
FONT_SIZE_ID       = 16  # Sequence ID font size
FONT_SIZE_TSD_BOX  = 16  # TSD box text font size
FONT_SIZE_TSD_SEQ  = 16  # TSD sequence font size
FONT_SIZE_TIR_BOX  = 16  # TIR box text font size
FONT_SIZE_TIR_SEQ  = 16  # TIR sequence font size
FONT_SIZE_EPVE_BOX = 16  # EPVE box text font size
FONT_SIZE_GAP      = 16  # Gap text font size

# Block dimensions and vertical offsets relative to baseline (y_ptr)
HEIGHT_TSD         = 1.0    # TSD block height
OFFSET_Y_TSD       = -0.5   # TSD block vertical offset
OFFSET_Y_TSD_SEQ   = -1.05  # TSD sequence text vertical offset

HEIGHT_TIR         = 1.0    # TIR block height
OFFSET_Y_TIR       = -0.5   # TIR block vertical offset
OFFSET_Y_TIR_SEQ   = 1.0    # TIR sequence text vertical offset

HEIGHT_EPVE        = 1.0    # EPVE block height
OFFSET_Y_EPVE      = -0.5   # EPVE block vertical offset

OFFSET_Y_GAP_TEXT  = -0.95  # Gap text vertical offset

# Line and layout settings
LINE_WIDTH_GAP     = 2.0    # Gap line width
Y_STEP             = 3.5    # Vertical spacing between entries

# ==========================================
# Data Parsing Logic
# ==========================================
def parse_evidence_file(file_path):
    parsed_data = []
    current_data = None
    
    if not os.path.exists(file_path):
        return parsed_data

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if "File:" in line:
                if current_data is not None:
                    parsed_data.append(current_data)
                
                raw_name = re.search(r"File:\s*(.*?)\.bed", line)
                if raw_name:
                    full_name = raw_name.group(1)
                    parts = full_name.split('_')
                    clean_id = "_".join(parts[3:]) if len(parts) > 3 else full_name
                else:
                    clean_id = "Unknown"

                current_data = {
                    "id": clean_id,
                    "epve_len": 0,
                    "tir_len": 0, "l_tir_seq": "", "r_tir_seq": "",
                    "tsd_len": 0, "tsd_seq": "",
                    "gaps": [0, 0, 0, 0]
                }
                continue
            
            if current_data is None: continue

            # EPVE length
            if "[EPVE" in line:
                m = re.search(r":(\d+)-(\d+)", line)
                if m: current_data["epve_len"] = abs(int(m.group(2)) - int(m.group(1)))
            
            # TIR (Terminal Inverted Repeat) length
            elif "[末端リピート]" in line:
                m = re.search(r"長さ:\s*(\d+)bp", line)
                if m: current_data["tir_len"] = int(m.group(1))
            
            # TIR sequence
            elif "Left TIR" in line:
                try:
                    next_line = next(f, "")
                    m = re.search(r"\[配列\]\s*([A-Za-z]+)", next_line)
                    if m: current_data["l_tir_seq"] = m.group(1).upper()
                except StopIteration:
                    pass
            
            elif "Right TIR" in line:
                try:
                    next_line = next(f, "")
                    m = re.search(r"\[配列\]\s*([A-Za-z]+)", next_line)
                    if m: current_data["r_tir_seq"] = m.group(1).upper()
                except StopIteration:
                    pass
            
            # TSD length
            elif "[TSD]" in line:
                m_len = re.search(r"長さ:\s*(\d+)bp", line)
                if m_len: current_data["tsd_len"] = int(m_len.group(1))
            
            # TSD sequence
            elif "Left TSD" in line:
                m_seq = re.search(r"\[配列\]\s*([A-Za-z]+)", line)
                if m_seq: current_data["tsd_seq"] = m_seq.group(1).upper()
                
            # Distance information (Gaps)
            elif "TSD-'" in line:
                m = re.search(r"TSD-'(\d+)'-TIR-'(\d+)'-EVE-'(\d+)'-TIR-'(\d+)'-TSD", line)
                if m: current_data["gaps"] = [int(x) for x in m.groups()]

        if current_data is not None:
            parsed_data.append(current_data)
            
    return parsed_data

# ==========================================
# Drawing Logic
# ==========================================
def draw_structure(parsed_data):
    if not parsed_data: return

    fig, ax = plt.subplots(figsize=(20, 3 * len(parsed_data)))
    y_ptr = 0
    max_x = 0

    for item in parsed_data:
        x = 0
        ax.text(-5, y_ptr, item['id'], ha='right', va='center', fontsize=FONT_SIZE_ID, fontweight='bold')
        
        # 1. Left TSD
        ax.add_patch(patches.Rectangle((x, y_ptr + OFFSET_Y_TSD), item['tsd_len'], HEIGHT_TSD, facecolor=COLOR_TSD, ec=COLOR_EDGE))
        ax.text(x + item['tsd_len']/2, y_ptr, f"TSD\n{item['tsd_len']} bps", ha='center', va='center', fontsize=FONT_SIZE_TSD_BOX, color=COLOR_TEXT_WHITE, fontweight='bold')
        ax.text(x + item['tsd_len']/2, y_ptr + OFFSET_Y_TSD_SEQ, item['tsd_seq'], ha='center', va='top', fontsize=FONT_SIZE_TSD_SEQ, color=COLOR_TSD, family='monospace', fontweight='bold')
        x += item['tsd_len']
        
        # Gap 1
        g1 = item['gaps'][0]
        dg1 = g1 / COMPRESS_RATE_GAP
        ax.plot([x, x + dg1], [y_ptr, y_ptr], color=COLOR_LINE, lw=LINE_WIDTH_GAP)
        ax.text(x + dg1/2, y_ptr + OFFSET_Y_GAP_TEXT, f"{g1} bps", ha='center', va='bottom', fontsize=FONT_SIZE_GAP)
        x += dg1
        
        # 2. Left TIR
        ax.add_patch(patches.Rectangle((x, y_ptr + OFFSET_Y_TIR), item['tir_len'], HEIGHT_TIR, facecolor=COLOR_TIR, ec=COLOR_EDGE))
        ax.text(x + item['tir_len']/2, y_ptr, f"L-TIR\n{item['tir_len']} bps", ha='center', va='center', fontsize=FONT_SIZE_TIR_BOX, color=COLOR_TEXT_WHITE, fontweight='bold')
        ax.text(x + item['tir_len']/2, y_ptr + OFFSET_Y_TIR_SEQ, item['l_tir_seq'], ha='center', va='top', fontsize=FONT_SIZE_TIR_SEQ, color=COLOR_TIR, family='monospace', fontweight='bold')
        x += item['tir_len']
        
        # Gap 2
        g2 = item['gaps'][1]
        dg2 = g2 / COMPRESS_RATE_GAP
        ax.plot([x, x + dg2], [y_ptr, y_ptr], color=COLOR_LINE, lw=LINE_WIDTH_GAP)
        ax.text(x + dg2/2, y_ptr + OFFSET_Y_GAP_TEXT, f"{g2} bps", ha='center', va='bottom', fontsize=FONT_SIZE_GAP)
        x += dg2
        
        # 3. EPVE Region
        depve = item['epve_len'] / COMPRESS_RATE_EPVE
        ax.add_patch(patches.Rectangle((x, y_ptr + OFFSET_Y_EPVE), depve, HEIGHT_EPVE, facecolor=COLOR_EPVE, ec=COLOR_EDGE))
        ax.text(x + depve/2, y_ptr, f"EVE\n{item['epve_len']} bps", ha='center', va='center', color=COLOR_TEXT_WHITE, fontweight='bold', fontsize=FONT_SIZE_EPVE_BOX)
        x += depve
        
        # Gap 3
        g3 = item['gaps'][2]
        dg3 = g3 / COMPRESS_RATE_GAP
        ax.plot([x, x + dg3], [y_ptr, y_ptr], color=COLOR_LINE, lw=LINE_WIDTH_GAP)
        ax.text(x + dg3/2, y_ptr + OFFSET_Y_GAP_TEXT, f"{g3} bps", ha='center', va='bottom', fontsize=FONT_SIZE_GAP)
        x += dg3
        
        # 4. Right TIR
        ax.add_patch(patches.Rectangle((x, y_ptr + OFFSET_Y_TIR), item['tir_len'], HEIGHT_TIR, facecolor=COLOR_TIR, ec=COLOR_EDGE))
        ax.text(x + item['tir_len']/2, y_ptr, f"R-TIR\n{item['tir_len']} bps", ha='center', va='center', fontsize=FONT_SIZE_TIR_BOX, color=COLOR_TEXT_WHITE, fontweight='bold')
        ax.text(x + item['tir_len']/2, y_ptr + OFFSET_Y_TIR_SEQ, item['r_tir_seq'], ha='center', va='top', fontsize=FONT_SIZE_TIR_SEQ, color=COLOR_TIR, family='monospace', fontweight='bold')
        x += item['tir_len']
        
        # Gap 4
        g4 = item['gaps'][3]
        dg4 = g4 / COMPRESS_RATE_GAP
        ax.plot([x, x + dg4], [y_ptr, y_ptr], color=COLOR_LINE, lw=LINE_WIDTH_GAP)
        ax.text(x + dg4/2, y_ptr + OFFSET_Y_GAP_TEXT, f"{g4} bps", ha='center', va='bottom', fontsize=FONT_SIZE_GAP)
        x += dg4
        
        # 5. Right TSD
        ax.add_patch(patches.Rectangle((x, y_ptr + OFFSET_Y_TSD), item['tsd_len'], HEIGHT_TSD, facecolor=COLOR_TSD, ec=COLOR_EDGE))
        ax.text(x + item['tsd_len']/2, y_ptr, f"TSD\n{item['tsd_len']} bps", ha='center', va='center', fontsize=FONT_SIZE_TSD_BOX, color=COLOR_TEXT_WHITE, fontweight='bold')
        ax.text(x + item['tsd_len']/2, y_ptr + OFFSET_Y_TSD_SEQ, item['tsd_seq'], ha='center', va='top', fontsize=FONT_SIZE_TSD_SEQ, color=COLOR_TSD, family='monospace', fontweight='bold')
        x += item['tsd_len']
        
        if x > max_x:
            max_x = x
            
        y_ptr -= Y_STEP

    ax.set_xlim(-10, max_x + 10)
    ax.set_ylim(y_ptr + 1.5, 2.0)
    
    ax.axis('off')
    plt.tight_layout()
    plt.savefig("structure_result.png", dpi=300, bbox_inches='tight')
    plt.savefig("structure_result.svg", bbox_inches='tight')
    print("Figures successfully saved as 'structure_result.png' and 'structure_result.svg'.")
    plt.show()

if __name__ == "__main__":
    input_file = "./Strict_Evidence_Summary_TIR_rev.txt" 
    
    if not os.path.exists(input_file):
        print(f"Error: Specified input file '{input_file}' was not found.")
        print("Please check if the file path is correct and accessible.")
    else:
        data = parse_evidence_file(input_file)
        if data:
            print(f"Parsing complete: Successfully retrieved {len(data)} entry/entries.")
            draw_structure(data)
        else:
            print(f"Error: Target file exists, but failed to parse valid entries.")
