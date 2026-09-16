#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
from collections import defaultdict
from bisect import bisect_left, bisect_right

import matplotlib.pyplot as plt
import scipy.stats as stats

# ==============================================================================
# Settings & Parameters
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. Input SAM files
# ------------------------------------------------------------------------------
INPUT_SAM_FILES = [
    "EVE_counterpart_AF.sam",
    "EVE_counterpart_NC.sam"
]

# ------------------------------------------------------------------------------
# 2. Shared individual count landscape
# ------------------------------------------------------------------------------
WINDOW_SIZE = 5000       # 5 kb
WINDOW_STEP = 1000       # 1 kb

# ------------------------------------------------------------------------------
# 3. Cen/Sat annotation
#
# Important:
#   centromere_pericentromere_satellites.bed
#   is a RepeatMasker-derived satellite extraction BED and
#   is not used for distinguishing centromere/pericentromere.
#
#   CHM13v2.0 Cen/Sat v2.1 is used for 3-color classification.
# ------------------------------------------------------------------------------
CENSAT_BED_FILE = "chm13v2.0_censat_v2.1.bed"

# ------------------------------------------------------------------------------
# 4. Output directory
# ------------------------------------------------------------------------------
OUTPUT_DIR = "EVE_HiC_Supplementary_Figure"

# ------------------------------------------------------------------------------
# 5. Figure settings
#
# Designed so that the genome-wide figure can be easily fine-tuned later.
# ------------------------------------------------------------------------------
FIG_DPI = 900

# Make width long enough to avoid chromosome label overlap
LANDSCAPE_FIG_WIDTH = 24
LANDSCAPE_FIG_HEIGHT = 6

# Chromosome labels
CHROM_LABEL_FONTSIZE = 12
CHROM_LABEL_ROTATION = 45

# Axis labels
AXIS_LABEL_FONTSIZE = 14

# Title
TITLE_FONTSIZE = 18

# Legend
LEGEND_FONTSIZE = 14
LEGEND_MARKER_SCALE = 4.0  # Legend marker scale factor (relative to plot size)

# Plot spot size (scatter points)
LANDSCAPE_SPOT_SIZE = 10

# Line width
LANDSCAPE_LINEWIDTH = 1.0

# Grid
GRID_ALPHA = 0.25
GRID_LINEWIDTH = 0.6

# Statistical text box settings
STAT_TEXT_FONTSIZE = 11
STAT_TEXT_X = 0.015         # x position (Axes relative coordinates: 0.0-1.0)
STAT_TEXT_Y = 0.95          # y position (Axes relative coordinates: 0.0-1.0)
STAT_TEXT_VA = 'top'        # vertical alignment ('top', 'center', 'bottom')
STAT_TEXT_HA = 'left'       # horizontal alignment ('left', 'center', 'right')
STAT_BOX_ALPHA = 0.85       # Text box background opacity

# ------------------------------------------------------------------------------
# 6. Genome-wide 3 colors
#
# Changing colors here adjusts the entire figure.
# ------------------------------------------------------------------------------
COLOR_CENTROMERE = "crimson"
COLOR_PERICENTROMERE = "royalblue"
COLOR_OTHER = "lightgray"

# ------------------------------------------------------------------------------
# 7. Maximum shared region individual x position plot
# ------------------------------------------------------------------------------
ZOOM_FLANK = 10_000

ZOOM_FIG_WIDTH = 14
ZOOM_FIG_HEIGHT = 7.5

ZOOM_INDIVIDUAL_FONTSIZE = 8
ZOOM_LINEWIDTH = 2.0

# ------------------------------------------------------------------------------
# 8. Chromosome order
# ------------------------------------------------------------------------------
CHROMOSOME_ORDER = (
    [f"chr{i}" for i in range(1, 23)]
    + ["chrX", "chrY"]
)

# ------------------------------------------------------------------------------
# 9. CHM13v2.0 chromosome lengths
#
# Genome-wide X coordinates use these actual chromosome lengths.
# ------------------------------------------------------------------------------
CHROMOSOME_LENGTHS = {
    "chr1": 248387328,
    "chr2": 242696752,
    "chr3": 201105948,
    "chr4": 193574945,
    "chr5": 182045439,
    "chr6": 172126628,
    "chr7": 160567428,
    "chr8": 146259331,
    "chr9": 150617247,
    "chr10": 134758134,
    "chr11": 135127769,
    "chr12": 133324548,
    "chr13": 113566686,
    "chr14": 101161492,
    "chr15": 99753195,
    "chr16": 96330374,
    "chr17": 84276897,
    "chr18": 80542538,
    "chr19": 61707364,
    "chr20": 66210255,
    "chr21": 45090682,
    "chr22": 51324926,
    "chrX": 154259566,
    "chrY": 62460029
}

# ------------------------------------------------------------------------------
# 10. Inter-chromosome gap
#
# Setting to 0 results in coordinates concatenated according to actual chromosome lengths.
# Recommended as 0 for manuscript figures.
#
# If labels are hard to read, adjust FIG_WIDTH / label rotation first
# instead of this value.
# ------------------------------------------------------------------------------
CHROMOSOME_GAP = 0

# ==============================================================================
# Basic functions
# ==============================================================================

def extract_individual_id(qname):
    """
    Extract individual ID from QNAME.

    Current naming convention:
        chr13_HG00133.xxx
             ↓
        HG00133
    """
    try:
        after_underscore = qname.split('_', 1)[1]
        return after_underscore.split('.', 1)[0]
    except IndexError:
        return None


def get_ref_length_from_cigar(cigar_str):
    """
    Calculate length consumed on reference genome from CIGAR string.
    """

    if not cigar_str or cigar_str == "*":
        return 0

    ops = re.findall(r'(\d+)([MIDNSH P=X])'.replace(" ", ""), cigar_str)

    ref_len = 0

    for count, op in ops:
        if op in ['M', 'D', 'N', '=', 'X']:
            ref_len += int(count)

    return ref_len


def chromosome_sort_key(chrom):
    """
    Sort in order: chr1, chr2, ..., chr22, chrX, chrY.
    """

    if chrom.startswith("chr"):
        value = chrom[3:]

        if value.isdigit():
            return int(value)

        if value == "X":
            return 23

        if value == "Y":
            return 24

    return 1000

# ==============================================================================
# Cen/Sat v2.1 annotation
# ==============================================================================

def classify_censat_feature(feature_name):
    """
    Aggregate CHM13 Cen/Sat v2.1 annotation names into 2 categories:

        centromere
        pericentromere

    Finally, regions not overlapping Cen/Sat are treated as "other"
    on the genome-wide plot.
    """

    name = feature_name.strip().lower()

    # ------------------------------------------------------------------
    # Centromere
    #
    # Alpha-satellite HOR systems
    #   hor
    #   dhor
    # Treat as centromere.
    # ------------------------------------------------------------------
    if (
        name.startswith("hor")
        or name.startswith("dhor")
    ):
        return "centromere"

    # ------------------------------------------------------------------
    # Pericentromere
    #
    # Group other satellite / transition features included in
    # centromeric/pericentromeric context in CHM13 Cen/Sat v2.1.
    # ------------------------------------------------------------------
    pericentromere_keywords = (
        "mon",
        "hsat",
        "bsat",
        "gsat",
        "censat",
        "ct"
    )

    if any(
        keyword in name
        for keyword in pericentromere_keywords
    ):
        return "pericentromere"

    # --------------------------------------------------------------
    # Unclassified
    # --------------------------------------------------------------
    return None


def load_censat_annotations(censat_bed_file):
    """
    Load CHM13v2.0 Cen/Sat v2.1 BED and store in format:

        reads_by_chrom[chrom] = [
            {
                'start': ...,
                'end': ...,
                'category': 'centromere' or 'pericentromere',
                'name': ...
            },
            ...
        ]

    BED uses 0-based half-open coordinates.
    """

    if not os.path.exists(censat_bed_file):
        raise FileNotFoundError(
            f"Cen/Sat annotation file not found: "
            f"{censat_bed_file}\n\n"
            f"Please place CHM13v2.0 Cen/Sat v2.1 BED file."
        )

    annotations_by_chrom = defaultdict(list)

    total_records = 0
    retained_records = 0

    with open(
        censat_bed_file,
        'r',
        encoding='utf-8'
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            if line.startswith('#'):
                continue

            if line.startswith('track'):
                continue

            if line.startswith('browser'):
                continue

            cols = line.split('\t')

            if len(cols) < 4:
                continue

            chrom = cols[0]

            try:
                start = int(cols[1])
                end = int(cols[2])
            except ValueError:
                continue

            name = cols[3]

            total_records += 1

            category = classify_censat_feature(
                name
            )

            if category is None:
                continue

            annotations_by_chrom[chrom].append({
                'start': start,
                'end': end,
                'category': category,
                'name': name
            })

            retained_records += 1

    # Sorted by coordinates per chromosome
    for chrom in annotations_by_chrom:

        annotations_by_chrom[chrom].sort(
            key=lambda x: (
                x['start'],
                x['end']
            )
        )

    print(
        f"  [CenSat] total annotation records: "
        f"{total_records:,}"
    )

    print(
        f"  [CenSat] centromere/pericentromere records retained: "
        f"{retained_records:,}"
    )

    return annotations_by_chrom


def get_window_censat_category(
    chrom,
    window_start,
    window_end,
    annotations_by_chrom
):
    """
    Determine which category the 5-kb window belongs to:

        centromere
        pericentromere
        other

    If a window spans multiple categories, adopt the category
    with the maximum overlap bp inside the window.

    In case of a tie:
        Prioritize centromere.

    Note:
        window_start/window_end are 0-based half-open.
    """

    annotations = annotations_by_chrom.get(
        chrom,
        []
    )

    if not annotations:
        return "other"

    centromere_overlap = 0
    pericentromere_overlap = 0

    for ann in annotations:

        if ann['end'] <= window_start:
            continue

        if ann['start'] >= window_end:
            break

        overlap_start = max(
            window_start,
            ann['start']
        )

        overlap_end = min(
            window_end,
            ann['end']
        )

        if overlap_end <= overlap_start:
            continue

        overlap = (
            overlap_end
            - overlap_start
        )

        if ann['category'] == "centromere":
            centromere_overlap += overlap

        elif ann['category'] == "pericentromere":
            pericentromere_overlap += overlap

    if (
        centromere_overlap == 0
        and pericentromere_overlap == 0
    ):
        return "other"

    if (
        centromere_overlap
        >= pericentromere_overlap
    ):
        return "centromere"

    return "pericentromere"

# ==============================================================================
# SAM Analysis
# ==============================================================================

def parse_sam_file(sam_path):
    """
    Parse SAM file.

    Important:
    - Keep all multi-mapping alignments from -k N
    - Use a set of ind_id when counting individuals
    - Do not interpret MAPQ 255 etc. as "unique mapping"
    """

    reads_by_chrom = defaultdict(list)
    all_individuals = set()

    # Diagnostic statistics
    total_mapped_alignments = 0

    alignments_per_individual = defaultdict(int)
    alignments_per_chrom = defaultdict(int)
    alignments_per_qname = defaultdict(int)

    qname_to_individual = {}

    if not os.path.exists(sam_path):
        print(f"Error: File not found -> {sam_path}")
        return None, None, None

    with open(sam_path, 'r', encoding='utf-8') as f:

        for line in f:

            if line.startswith('@'):
                continue

            cols = line.rstrip('\n').split('\t')

            if len(cols) < 11:
                continue

            try:
                qname = cols[0]
                flag = int(cols[1])
                rname = cols[2]
                pos = int(cols[3])
                mapq = int(cols[4])
                cigar = cols[5]
                seq = cols[9]
            except (ValueError, IndexError):
                continue

            # ------------------------------------------------------------------
            # Exclude unmapped
            # ------------------------------------------------------------------
            if flag & 4 or rname == '*':
                continue

            # ------------------------------------------------------------------
            # Individual ID
            # ------------------------------------------------------------------
            ind_id = extract_individual_id(qname)

            if not ind_id:
                continue

            # ------------------------------------------------------------------
            # Reference length from CIGAR
            # ------------------------------------------------------------------
            ref_len = get_ref_length_from_cigar(cigar)

            if ref_len == 0:
                ref_len = len(seq) if seq != '*' else 1

            end_pos = pos + ref_len - 1

            # ------------------------------------------------------------------
            # Register individual
            # ------------------------------------------------------------------
            all_individuals.add(ind_id)

            if qname in qname_to_individual:

                if qname_to_individual[qname] != ind_id:
                    print(
                        f"Warning: Different individual IDs detected from the same QNAME: "
                        f"{qname} -> "
                        f"{qname_to_individual[qname]} / {ind_id}"
                    )

            else:
                qname_to_individual[qname] = ind_id

            # ------------------------------------------------------------------
            # Save alignment
            # ------------------------------------------------------------------
            reads_by_chrom[rname].append({
                'qname': qname,
                'ind': ind_id,
                'start': pos,
                'end': end_pos,
                'mapq': mapq,
                'flag': flag,
                'cigar': cigar
            })

            # ------------------------------------------------------------------
            # Diagnostic statistics
            # ------------------------------------------------------------------
            total_mapped_alignments += 1
            alignments_per_individual[ind_id] += 1
            alignments_per_chrom[rname] += 1
            alignments_per_qname[qname] += 1

    mapped_qname_count = len(alignments_per_qname)

    multimapping_qname_count = sum(
        1
        for count in alignments_per_qname.values()
        if count > 1
    )

    max_alignments_per_qname = (
        max(alignments_per_qname.values())
        if alignments_per_qname
        else 0
    )

    qname_alignment_distribution = defaultdict(int)

    for count in alignments_per_qname.values():
        qname_alignment_distribution[count] += 1

    stats = {
        'total_mapped_alignments':
            total_mapped_alignments,

        'mapped_qname_count':
            mapped_qname_count,

        'multimapping_qname_count':
            multimapping_qname_count,

        'max_alignments_per_qname':
            max_alignments_per_qname,

        'alignments_per_individual':
            dict(alignments_per_individual),

        'alignments_per_chrom':
            dict(alignments_per_chrom),

        'qname_alignment_distribution':
            dict(qname_alignment_distribution)
    }

    return (
        reads_by_chrom,
        sorted(all_individuals),
        stats
    )


# ==============================================================================
# Original 'maximum shared region' logic
# ==============================================================================

def find_max_shared_region(reads_by_chrom, all_individuals, window_size):
    """
    Maintain current analysis logic and retrieve one region where
    "the number of independent individuals existing within window_size" is maximized.

    Multiple alignments from the same individual are treated as 1 individual.
    """

    if not all_individuals:
        return None

    best_region = None
    best_count = 0

    for chrom, reads in reads_by_chrom.items():

        if not reads:
            continue

        reads.sort(key=lambda x: (x['start'], x['end']))

        n = len(reads)

        for i in range(n):

            min_start = reads[i]['start']

            window_reads = []
            current_inds = set()
            max_end = min_start

            for j in range(i, n):

                r = reads[j]

                span = r['end'] - min_start + 1

                if span > window_size:
                    break

                window_reads.append(r)
                current_inds.add(r['ind'])

                if r['end'] > max_end:
                    max_end = r['end']

            if not window_reads:
                continue

            span = max_end - min_start + 1
            present_count = len(current_inds)

            is_better = False

            if best_region is None:

                is_better = True

            elif present_count > best_count:

                is_better = True

            elif present_count == best_count:

                if span < best_region['span']:

                    is_better = True

                elif (
                    span == best_region['span']
                    and min_start < best_region['start']
                ):

                    is_better = True

            if is_better:

                missing_inds = sorted(
                    set(all_individuals) - current_inds
                )

                best_region = {
                    'chrom': chrom,
                    'start': min_start,
                    'end': max_end,
                    'span': span,
                    'present_count': present_count,
                    'present_inds': sorted(current_inds),
                    'missing_inds': missing_inds,
                    'reads': list(window_reads)
                }

                best_count = present_count

    return best_region


# ==============================================================================
# For sliding window:
# Calculate number of individuals with alignments fully contained within window
# ==============================================================================

def build_sliding_window_landscape(
    reads_by_chrom,
    window_size,
    window_step,
    annotations_by_chrom
):
    """
    Build sliding windows across all chromosomes and store:

        1. Number of independent individuals present in window
        2. Category by Cen/Sat annotation

    category:
        centromere
        pericentromere
        other

    Important:
        Manage window coordinates in 0-based half-open format like BED.
    """

    landscape = []

    for chrom in sorted(
        reads_by_chrom.keys(),
        key=chromosome_sort_key
    ):

        if chrom not in CHROMOSOME_LENGTHS:
            print(
                f"  Warning: Skipping because chromosome length is undefined: "
                f"{chrom}"
            )
            continue

        reads = reads_by_chrom[chrom]

        if not reads:
            continue

        reads = sorted(
            reads,
            key=lambda x: (
                x['start'],
                x['end']
            )
        )

        chromosome_length = CHROMOSOME_LENGTHS[
            chrom
        ]

        # ------------------------------------------------------------------
        # First window
        # ------------------------------------------------------------------
        window_start = 0

        while window_start < chromosome_length:

            window_end = min(
                window_start + window_size,
                chromosome_length
            )

            individual_set = set()

            # ------------------------------------------------------------------
            # Alignments inside window
            #
            # Only adopt alignments fully contained within window.
            # ------------------------------------------------------------------
            for r in reads:

                if r['start'] < window_start:
                    continue

                if r['start'] >= window_end:
                    break

                if r['end'] < window_end:
                    individual_set.add(
                        r['ind']
                    )

            # ------------------------------------------------------------------
            # Cen/Sat class
            # ------------------------------------------------------------------
            category = get_window_censat_category(
                chrom,
                window_start,
                window_end,
                annotations_by_chrom
            )

            landscape.append({
                'chrom': chrom,
                'start': window_start,
                'end': window_end,
                'n_individuals': len(
                    individual_set
                ),
                'category': category
            })

            window_start += window_step

    return landscape

# ==============================================================================
# CSV Output
# ==============================================================================

def write_landscape_csv(dataset_name, landscape):
    """
    Save sliding window landscape as CSV.
    """

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{dataset_name}_5kb_sliding_landscape.csv"
    )

    with open(
        output_path,
        'w',
        encoding='utf-8',
        newline=''
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            'dataset',
            'chromosome',
            'window_start',
            'window_end',
            'window_size_bp',
            'window_step_bp',
            'number_of_individuals',
            'region_category'
        ])

        for row in landscape:

            writer.writerow([
                dataset_name,
                row['chrom'],
                row['start'],
                row['end'],
                WINDOW_SIZE,
                WINDOW_STEP,
                row['n_individuals'],
                row['category']
            ])

    return output_path


# ==============================================================================
# Coordinate transformation for genome-wide plot
# ==============================================================================

def make_genome_offsets():
    """
    Create genome-wide plotting offset using actual chromosome lengths of CHM13v2.0.

    When CHROMOSOME_GAP = 0,
    coordinates are concatenated according to actual lengths.
    """

    offsets = {}

    current_offset = 0

    for chrom in CHROMOSOME_ORDER:

        if chrom not in CHROMOSOME_LENGTHS:
            continue

        offsets[chrom] = current_offset

        current_offset += (
            CHROMOSOME_LENGTHS[chrom]
            + CHROMOSOME_GAP
        )

    total_genome_plot_length = current_offset

    return (
        offsets,
        total_genome_plot_length
    )

# ==============================================================================
# Genome-wide landscape plot
# ==============================================================================

def plot_genome_wide_landscape(
    dataset_name,
    landscape,
    best_region,
    total_individuals
):
    """
    Genome-wide landscape for manuscript Supplementary Figure.

    3 colors:
        centromere
        pericentromere
        other

    y-axis:
        Number of independent individuals in 5-kb window

    x-axis:
        Based on actual chromosome lengths of CHM13v2.0.
    """

    offsets, total_genome_plot_length = (
        make_genome_offsets()
    )

    fig, ax = plt.subplots(
        figsize=(
            LANDSCAPE_FIG_WIDTH,
            LANDSCAPE_FIG_HEIGHT
        )
    )

    # ------------------------------------------------------------------
    # Plot by category
    # ------------------------------------------------------------------
    category_colors = {
        "centromere":
            COLOR_CENTROMERE,

        "pericentromere":
            COLOR_PERICENTROMERE,

        "other":
            COLOR_OTHER
    }

    category_labels = {
        "centromere":
            "Centromere",

        "pericentromere":
            "Pericentromere",

        "other":
            "Other"
    }

    # Control overlap order (zorder): Display Centromere on top
    category_zorder = {
        "other": 1,
        "pericentromere": 2,
        "centromere": 3
    }

    # Call plotting in legend order (Centromere -> Pericentromere -> Other)
    for category in (
        "centromere",
        "pericentromere",
        "other"
    ):

        x_values = []
        y_values = []

        for row in landscape:

            if row['category'] != category:
                continue

            # Exclude windows with Y = 0 (0 shared individuals) from plotting
            if row['n_individuals'] == 0:
                continue

            if row['chrom'] not in offsets:
                continue

            # window center
            window_center = (
                row['start']
                + row['end']
            ) / 2

            x = (
                offsets[row['chrom']]
                + window_center
            )

            x_values.append(x)

            y_values.append(
                row['n_individuals']
            )

        if not x_values:
            continue

        ax.scatter(
            x_values,
            y_values,
            s=LANDSCAPE_SPOT_SIZE,
            color=category_colors[
                category
            ],
            label=category_labels[
                category
            ],
            zorder=category_zorder[
                category
            ],
            linewidths=0
        )

    # ------------------------------------------------------------------
    # Chromosome boundaries
    # ------------------------------------------------------------------
    for chrom in CHROMOSOME_ORDER:

        if chrom not in offsets:
            continue

        x_boundary = offsets[chrom]

        if x_boundary == 0:
            continue

        ax.axvline(
            x_boundary,
            linewidth=0.45,
            alpha=0.25,
            color="black"
        )

    # ------------------------------------------------------------------
    # Highlight maximum shared region
    # ------------------------------------------------------------------
    if best_region is not None:

        best_chrom = best_region['chrom']

        if best_chrom in offsets:

            highlight_start = (
                offsets[best_chrom]
                + best_region['start']
            )

            highlight_end = (
                offsets[best_chrom]
                + best_region['end']
            )

            ax.axvspan(
                highlight_start,
                highlight_end,
                alpha=0.18,
                color="gold"
            )

            max_fraction = (
                best_region['present_count']
                / total_individuals
                * 100
                if total_individuals > 0
                else 0
            )

            ax.annotate(
                (
                    f"{best_chrom}:"
                    f"{best_region['start']:,}-"
                    f"{best_region['end']:,}\n"
                    f"{best_region['present_count']}/"
                    f"{total_individuals} "
                    f"({max_fraction:.1f}%)"
                ),
                xy=(
                    (
                        highlight_start
                        + highlight_end
                    ) / 2,
                    best_region[
                        'present_count'
                    ]
                ),
                xytext=(
                    0,
                    25
                ),
                textcoords='offset points',
                ha='center',
                va='bottom',
                fontsize=9,
                arrowprops={
                    'arrowstyle': '->',
                    'linewidth': 0.8
                }
            )

    # ------------------------------------------------------------------
    # X-axis: Actual chromosome lengths
    # ------------------------------------------------------------------
    tick_positions = []
    tick_labels = []

    for chrom in CHROMOSOME_ORDER:

        if chrom not in offsets:
            continue

        center = (
            offsets[chrom]
            + CHROMOSOME_LENGTHS[chrom] / 2
        )

        tick_positions.append(center)
        tick_labels.append(chrom)

    ax.set_xticks(
        tick_positions
    )

    ax.set_xticklabels(
        tick_labels,
        fontsize=CHROM_LABEL_FONTSIZE,
        rotation=CHROM_LABEL_ROTATION,
        ha='center',
        va='top'
    )

    # ------------------------------------------------------------------
    # Y-axis
    # ------------------------------------------------------------------
    max_y = max(
        row['n_individuals']
        for row in landscape
    ) if landscape else 0

    ax.set_ylim(
        0,
        max(
            total_individuals,
            max_y + 1
        )
    )

    ax.set_ylabel(
        "Number of independent individuals",
        fontsize=AXIS_LABEL_FONTSIZE
    )

    ax.set_xlabel(
        "CHM13v2.0 genomic position",
        fontsize=AXIS_LABEL_FONTSIZE
    )

    # ------------------------------------------------------------------
    # Title
    # ------------------------------------------------------------------
    ax.set_title(
        (
            f"{dataset_name}: EVE-counterpart mapping landscape\n"
            f"{WINDOW_SIZE / 1000:.0f}-kb window, "
            f"{WINDOW_STEP / 1000:.0f}-kb sliding step"
        ),
        fontsize=TITLE_FONTSIZE
    )

    # ------------------------------------------------------------------
    # Grid
    # ------------------------------------------------------------------
    ax.grid(
        axis='y',
        alpha=GRID_ALPHA,
        linewidth=GRID_LINEWIDTH
    )

    # ------------------------------------------------------------------
    # Statistical test (Mann-Whitney U test: Cen/Peri vs Other)
    # ------------------------------------------------------------------
    # To prevent false positives due to sliding window overlap,
    # extract only independent non-overlapping windows every 5 kb for testing
    independent_landscape = [
        row for row in landscape 
        if row['start'] % WINDOW_SIZE == 0
    ]
    
    cen_peri_counts = [
        row['n_individuals'] for row in independent_landscape 
        if row['category'] in ('centromere', 'pericentromere')
    ]
    
    other_counts = [
        row['n_individuals'] for row in independent_landscape 
        if row['category'] == 'other'
    ]

    if cen_peri_counts and other_counts:
        # One-sided test: Whether shared individual counts in Cen/Peri are greater than Other
        stat, pval = stats.mannwhitneyu(
            cen_peri_counts, 
            other_counts, 
            alternative='greater'
        )
        
        if pval == 0.0:
            p_str = "p < 1e-300"
        elif pval < 1e-4:
            p_str = f"p = {pval:.2e}"
        elif pval < 0.001:
            p_str = "p < 0.001"
        else:
            p_str = f"p = {pval:.4f}"
            
        stat_text = (
            "Mann-Whitney U test\n"
            "(Cen/Peri vs. Other):\n"
            f"{p_str}"
        )
        
        # Place statistical text box
        ax.text(
            STAT_TEXT_X, STAT_TEXT_Y, 
            stat_text,
            transform=ax.transAxes,
            fontsize=STAT_TEXT_FONTSIZE,
            verticalalignment=STAT_TEXT_VA,
            horizontalalignment=STAT_TEXT_HA,
            bbox=dict(
                boxstyle='round,pad=0.5',
                facecolor='white',
                alpha=STAT_BOX_ALPHA,
                edgecolor='gray',
                linewidth=0.8
            ),
            zorder=5
        )

    # ------------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------------
    ax.legend(
        loc='upper right',
        fontsize=LEGEND_FONTSIZE,
        markerscale=LEGEND_MARKER_SCALE,
        frameon=True
    )
    # ------------------------------------------------------------------
    # X limits
    # ------------------------------------------------------------------
    ax.set_xlim(
        0,
        total_genome_plot_length
    )

    fig.subplots_adjust(
        left=0.06,
        right=0.995,
        top=0.88,
        bottom=0.24
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    png_path = os.path.join(
        OUTPUT_DIR,
        f"{dataset_name}_genomewide_landscape.png"
    )

    pdf_path = os.path.join(
        OUTPUT_DIR,
        f"{dataset_name}_genomewide_landscape.pdf"
    )

    fig.savefig(
        png_path,
        dpi=FIG_DPI,
        bbox_inches='tight'
    )

    fig.savefig(
        pdf_path,
        bbox_inches='tight'
    )

    plt.close(fig)

    return (
        png_path,
        pdf_path
    )

# ==============================================================================
# Individual x position plot for maximum shared region
# ==============================================================================

def plot_best_region_individual_tracks(
    dataset_name,
    best_region,
    all_individuals
):
    """
    Plot maximum shared region as individual x position.

    Display all individuals and draw segments only for individuals with alignments in this region.

    Multiple-mapping alignments of the same individual are all displayed in that individual's row.
    """

    if best_region is None:
        return None, None

    chrom = best_region['chrom']

    # ------------------------------------------------------------------
    # Display range
    # ------------------------------------------------------------------
    region_start = max(
        1,
        best_region['start'] - ZOOM_FLANK
    )

    region_end = (
        best_region['end']
        + ZOOM_FLANK
    )

    # Individual order
    individuals = sorted(all_individuals)

    y_positions = {
        ind: idx
        for idx, ind in enumerate(individuals)
    }

    fig, ax = plt.subplots(
        figsize=(ZOOM_FIG_WIDTH, ZOOM_FIG_HEIGHT)
    )

    # ------------------------------------------------------------------
    # Display maximum shared window as background
    # ------------------------------------------------------------------
    ax.axvspan(
        best_region['start'],
        best_region['end'],
        alpha=0.20
    )

    # ------------------------------------------------------------------
    # Plot alignments per individual
    # ------------------------------------------------------------------
    for ind in individuals:

        y = y_positions[ind]

        ind_reads = [
            r
            for r in best_region['reads']
            if r['ind'] == ind
        ]

        # Draw nothing if the individual is not present
        for r in ind_reads:

            ax.plot(
                [r['start'], r['end']],
                [y, y],
                linewidth=2.0,
                solid_capstyle='butt'
            )

    # ------------------------------------------------------------------
    # Y-axis
    # ------------------------------------------------------------------
    ax.set_yticks(
        list(y_positions.values())
    )

    ax.set_yticklabels(
        individuals,
        fontsize=8
    )

    ax.invert_yaxis()

    # ------------------------------------------------------------------
    # X-axis
    # ------------------------------------------------------------------
    ax.set_xlim(
        region_start,
        region_end
    )

    ax.set_xlabel(
        f"{chrom} genomic position (bp)"
    )

    ax.set_ylabel(
        "Individual"
    )

    ax.set_title(
        (
            f"{dataset_name}: maximum shared region\n"
            f"{chrom}:{best_region['start']:,}-"
            f"{best_region['end']:,} "
            f"({best_region['span']:,} bp), "
            f"{best_region['present_count']}/"
            f"{len(all_individuals)} individuals"
        )
    )

    ax.grid(
        axis='x',
        alpha=0.25,
        linewidth=0.6
    )

    fig.tight_layout()

    png_path = os.path.join(
        OUTPUT_DIR,
        f"{dataset_name}_maximum_region_individual_tracks.png"
    )

    pdf_path = os.path.join(
        OUTPUT_DIR,
        f"{dataset_name}_maximum_region_individual_tracks.pdf"
    )

    fig.savefig(
        png_path,
        dpi=FIG_DPI,
        bbox_inches='tight'
    )

    fig.savefig(
        pdf_path,
        bbox_inches='tight'
    )

    plt.close(fig)

    return png_path, pdf_path


# ==============================================================================
# Save maximum shared region information
# ==============================================================================

def write_best_region_report(
    dataset_name,
    best_region,
    all_individuals,
    stats
):
    """
    Save maximum shared region and diagnostic information as text.
    """

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{dataset_name}_maximum_region_report.txt"
    )

    total_individuals = len(all_individuals)

    with open(
        output_path,
        'w',
        encoding='utf-8'
    ) as out:

        out.write("=" * 90 + "\n")
        out.write(
            f" EVE Hi-C Supplementary Figure analysis: {dataset_name}\n"
        )
        out.write("=" * 90 + "\n\n")

        out.write(
            f"Detected individuals: "
            f"{total_individuals}\n"
        )

        out.write(
            f"Window size: "
            f"{WINDOW_SIZE:,} bp\n"
        )

        out.write(
            f"Sliding step: "
            f"{WINDOW_STEP:,} bp\n\n"
        )

        # ------------------------------------------------------------------
        # Diagnostic information
        # ------------------------------------------------------------------
        out.write(
            "[SAM diagnostic information]\n"
        )

        out.write(
            f"Mapped alignments: "
            f"{stats['total_mapped_alignments']:,}\n"
        )

        out.write(
            f"Mapped QNAMEs: "
            f"{stats['mapped_qname_count']:,}\n"
        )

        out.write(
            f"Multi-mapping QNAMEs: "
            f"{stats['multimapping_qname_count']:,}\n"
        )

        out.write(
            f"Maximum alignments/QNAME: "
            f"{stats['max_alignments_per_qname']:,}\n"
        )

        if stats['mapped_qname_count']:

            fraction = (
                stats['multimapping_qname_count']
                / stats['mapped_qname_count']
                * 100
            )

        else:

            fraction = 0.0

        out.write(
            f"Multi-mapping QNAME fraction: "
            f"{fraction:.2f}%\n\n"
        )

        # ------------------------------------------------------------------
        # Maximum region
        # ------------------------------------------------------------------
        if best_region is None:

            out.write(
                "No maximum shared region was detected.\n"
            )

            return output_path

        reg = best_region

        shared_fraction = (
            reg['present_count']
            / total_individuals
            * 100
            if total_individuals > 0
            else 0.0
        )

        out.write(
            "[Maximum shared region]\n"
        )

        out.write(
            f"Chromosome: "
            f"{reg['chrom']}\n"
        )

        out.write(
            f"Start: "
            f"{reg['start']:,}\n"
        )

        out.write(
            f"End: "
            f"{reg['end']:,}\n"
        )

        out.write(
            f"Span: "
            f"{reg['span']:,} bp\n"
        )

        out.write(
            f"Shared individuals: "
            f"{reg['present_count']} / "
            f"{total_individuals} "
            f"({shared_fraction:.2f}%)\n"
        )

        out.write(
            "Present individuals:\n"
        )

        for ind in reg['present_inds']:
            out.write(
                f"  {ind}\n"
            )

        out.write(
            "\nMissing individuals:\n"
        )

        for ind in reg['missing_inds']:
            out.write(
                f"  {ind}\n"
            )

        out.write(
            f"\nAlignments in maximum region: "
            f"{len(reg['reads']):,}\n"
        )

    return output_path

# ==============================================================================
# Individual Analysis
# ==============================================================================

def analyze_dataset(sam_path, dataset_name):

    print()
    print("=" * 90)
    print(f"  {dataset_name} ANALYSIS")
    print("=" * 90)

    reads_by_chrom, all_individuals, stats = parse_sam_file(
        sam_path
    )

    if reads_by_chrom is None:
        return None

    total_individuals = len(all_individuals)

    # ------------------------------------------------------------------
    # CHM13 Cen/Sat v2.1 annotation
    # ------------------------------------------------------------------
    annotations_by_chrom = load_censat_annotations(
        CENSAT_BED_FILE
    )

    print(
        f"  mapped alignments : "
        f"{stats['total_mapped_alignments']:,}"
    )

    print(
        f"  mapped QNAMEs     : "
        f"{stats['mapped_qname_count']:,}"
    )

    print(
        f"  multi-mapping QNAMEs : "
        f"{stats['multimapping_qname_count']:,}"
    )

    print(
        f"  max alignments/QNAME : "
        f"{stats['max_alignments_per_qname']:,}"
    )

    print(
        f"  detected individuals : "
        f"{total_individuals}"
    )

    # ------------------------------------------------------------------
    # Original maximum shared region
    # ------------------------------------------------------------------
    print()
    print(
        "  [1] maximum shared region search ..."
    )

    best_region = find_max_shared_region(
        reads_by_chrom,
        all_individuals,
        WINDOW_SIZE
    )

    if best_region is not None:

        fraction = (
            best_region['present_count']
            / total_individuals
            * 100
        )

        print(
            f"      maximum = "
            f"{best_region['present_count']}/"
            f"{total_individuals} "
            f"({fraction:.2f}%)"
        )

        print(
            f"      region = "
            f"{best_region['chrom']}:"
            f"{best_region['start']:,}-"
            f"{best_region['end']:,}"
        )

    else:

        print(
            "      no maximum shared region"
        )

    # ------------------------------------------------------------------
    # Sliding window landscape
    # ------------------------------------------------------------------
    print()
    print(
        "  [2] building sliding-window landscape ..."
    )

    landscape = build_sliding_window_landscape(
        reads_by_chrom,
        WINDOW_SIZE,
        WINDOW_STEP,
        annotations_by_chrom
    )

    print(
        f"      windows = "
        f"{len(landscape):,}"
    )

    max_landscape_individuals = max(
        row['n_individuals']
        for row in landscape
    ) if landscape else 0

    print(
        f"      maximum individuals in sliding windows = "
        f"{max_landscape_individuals}/"
        f"{total_individuals}"
    )

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------
    csv_path = write_landscape_csv(
        dataset_name,
        landscape
    )

    print(
        f"      CSV = {csv_path}"
    )

    # ------------------------------------------------------------------
    # Maximum region report
    # ------------------------------------------------------------------
    report_path = write_best_region_report(
        dataset_name,
        best_region,
        all_individuals,
        stats
    )

    print(
        f"      report = {report_path}"
    )

    # ------------------------------------------------------------------
    # Genome-wide figure
    # ------------------------------------------------------------------
    png1, pdf1 = plot_genome_wide_landscape(
        dataset_name,
        landscape,
        best_region,
        total_individuals
    )

    print(
        f"      genome-wide PNG = {png1}"
    )

    print(
        f"      genome-wide PDF = {pdf1}"
    )

    # ------------------------------------------------------------------
    # Maximum region individual tracks
    # ------------------------------------------------------------------
    if best_region is not None:

        png2, pdf2 = plot_best_region_individual_tracks(
            dataset_name,
            best_region,
            all_individuals
        )

        print(
            f"      zoom PNG = {png2}"
        )

        print(
            f"      zoom PDF = {pdf2}"
        )

    else:

        png2 = None
        pdf2 = None

    return {
        'dataset_name': dataset_name,
        'reads_by_chrom': reads_by_chrom,
        'all_individuals': all_individuals,
        'total_individuals': total_individuals,
        'stats': stats,
        'landscape': landscape,
        'best_region': best_region,
        'genomewide_png': png1,
        'genomewide_pdf': pdf1,
        'zoom_png': png2,
        'zoom_pdf': pdf2
    }


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    results = {}

    # --------------------------------------------------------------------------
    # AF
    # --------------------------------------------------------------------------
    if os.path.exists(
        "EVE_counterpart_AF.sam"
    ):

        results['AF'] = analyze_dataset(
            "EVE_counterpart_AF.sam",
            "AF"
        )

    else:

        print(
            "Warning: EVE_counterpart_AF.sam not found."
        )

    # --------------------------------------------------------------------------
    # NC
    # --------------------------------------------------------------------------
    if os.path.exists(
        "EVE_counterpart_NC.sam"
    ):

        results['NC'] = analyze_dataset(
            "EVE_counterpart_NC.sam",
            "NC"
        )

    else:

        print(
            "Warning: EVE_counterpart_NC.sam not found."
        )

    # --------------------------------------------------------------------------
    # Final summary
    # --------------------------------------------------------------------------
    print()
    print("=" * 90)
    print("  ANALYSIS COMPLETE")
    print("=" * 90)

    for dataset_name, result in results.items():

        if result is None:
            continue

        best = result['best_region']

        print()
        print(
            f"[{dataset_name}]"
        )

        print(
            f"  individuals = "
            f"{result['total_individuals']}"
        )

        if best is not None:

            fraction = (
                best['present_count']
                / result['total_individuals']
                * 100
            )

            print(
                f"  maximum shared region = "
                f"{best['chrom']}:"
                f"{best['start']:,}-"
                f"{best['end']:,}"
            )

            print(
                f"  shared individuals = "
                f"{best['present_count']}/"
                f"{result['total_individuals']} "
                f"({fraction:.2f}%)"
            )

    print()
    print(
        f"All outputs are in: {OUTPUT_DIR}/"
    )
