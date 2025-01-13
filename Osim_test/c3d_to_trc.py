# -*- coding: utf-8 -*-
"""

@author: Maksim Krivolapov

@mailto: mkrivolapov@darimotion.com
    
version history:
    
"""

import ezc3d
import os

def c3d_to_trc_ezc3d(input_c3d_path, output_trc_path):
    """
    Convert marker data from a C3D file to a TRC file for OpenSim using the ezc3d library.
    """

    # --- 1) Load the C3D file ---
    c3d = ezc3d.c3d(input_c3d_path)

    # Extract marker labels (list of strings)
    marker_labels = c3d["parameters"]["POINT"]["LABELS"]["value"]
    num_markers = len(marker_labels)

    # 3D marker data: shape is (4, num_markers, num_frames)
    #   - [0, :, :] = X
    #   - [1, :, :] = Y
    #   - [2, :, :] = Z
    #   - [3, :, :] = residual/error
    marker_data = c3d["data"]["points"]

    num_frames = marker_data.shape[2]

    # Sampling rate (frame rate) of the marker data
    #  - Could come from c3d["header"]["frameRate"]
    #  - Or from c3d["parameters"]["POINT"]["RATE"]["value"][0]
    #    (depending on how your C3D is set up)
    if "RATE" in c3d["parameters"]["POINT"]:
        data_rate = c3d["parameters"]["POINT"]["RATE"]["value"][0]
    else:
        data_rate = c3d["header"]["frameRate"]
    camera_rate = data_rate  # Often identical for motion capture data

    # Time increment per frame
    dt = 1.0 / data_rate

    # --- 2) Prepare TRC file header info ---
    # TRC has a standard format recognized by OpenSim. Example:
    #
    # 1) PathFileType 4 (X/Y) TRC
    # 2) DataRate CameraRate NumFrames NumMarkers Units OrigDataRate OrigDataStartFrame OrigNumFrames
    # 3) Frame#  Time  M1X M1Y M1Z  M2X M2Y M2Z ...
    #
    # Then the per-frame data.

    # 1) First line
    header_line_1 = "PathFileType\t4\t(X/Y)\tTRC"

    # 2) Second line
    #    DataRate  CameraRate  NumFrames  NumMarkers  Units  OrigDataRate  OrigDataStartFrame  OrigNumFrames
    units = "mm"  # or "m", depending on your data
    orig_data_rate = data_rate
    orig_data_start_frame = 1
    orig_num_frames = num_frames

    header_line_2 = (
        f"{data_rate}\t{camera_rate}\t{num_frames}\t{num_markers}\t"
        f"{units}\t{orig_data_rate}\t{orig_data_start_frame}\t{orig_num_frames}"
    )

    # 3) Column labels
    #    Format: Frame#  Time  <marker1X> <marker1Y> <marker1Z>  <marker2X> ...
    label_line = "Frame#\tTime"
    for marker in marker_labels:
        label_line += f"\t{marker}_X\t{marker}_Y\t{marker}_Z"

    # --- 3) Write data rows ---
    # Each row: FrameIndex, Time, x1, y1, z1, x2, y2, z2, ...
    data_lines = []
    for f in range(num_frames):
        frame_num = f + 1
        time_sec = f * dt
        row = [str(frame_num), f"{time_sec:.5f}"]

        for m in range(num_markers):
            x = marker_data[0, m, f]
            y = marker_data[1, m, f]
            z = marker_data[2, m, f]
            row.extend([f"{x:.5f}", f"{y:.5f}", f"{z:.5f}"])

        data_lines.append("\t".join(row))

    # --- 4) Write out the TRC file ---
    with open(output_trc_path, "w") as f_out:
        f_out.write(header_line_1 + "\n")
        f_out.write(header_line_2 + "\n")
        f_out.write(label_line + "\n")
        for line in data_lines:
            f_out.write(line + "\n")

    print(f"TRC file saved to: {os.path.abspath(output_trc_path)}")

# Example usage (uncomment and adjust paths as needed):
if __name__ == "__main__":
     input_c3d = "Sub_1_bilateral_squat02_.centers.c3d"
     output_trc = "Sub_1_bilateral_squat02_.centers.trc"
     c3d_to_trc_ezc3d(input_c3d, output_trc)

