# -*- coding: utf-8 -*-
"""

@author: Maksim Krivolapov

@mailto: mkrivolapov@darimotion.com
    
version history:
    
"""

import ezc3d
import numpy as np
import pandas as pd
import os
import json
from io import StringIO

#-----------------------------------------------------------------------------#
def resample_marker_data(marker_data, old_rate, new_rate):
    """
    Resample marker_data (shape: [4, num_markers, num_frames]) from old_rate (Hz)
    to new_rate (Hz) using 1D interpolation (np.interp).
    Returns (new_marker_data, new_num_frames).

    marker_data[0, m, f] = X
    marker_data[1, m, f] = Y
    marker_data[2, m, f] = Z
    marker_data[3, m, f] = residual
    """
    _, num_markers, old_num_frames = marker_data.shape

    # Total duration in seconds
    duration = (old_num_frames - 1) / old_rate

    # Number of frames at the new rate
    new_num_frames = int(round(duration * new_rate)) + 1

    # Time vectors
    old_time = np.linspace(0, duration, old_num_frames)
    new_time = np.linspace(0, duration, new_num_frames)

    # Prepare the resampled array
    new_marker_data = np.zeros((4, num_markers, new_num_frames))

    for dim in range(4):
        for m in range(num_markers):
            new_marker_data[dim, m, :] = np.interp(
                new_time, old_time, marker_data[dim, m, :]
            )

    return new_marker_data, new_num_frames


#-----------------------------------------------------------------------------#
def c3d_to_trc_custom(
    input_c3d_path, 
    output_trc_path, 
    new_rate=100.0,
    assume_z_up=True
):
    """
    Reads a C3D with ezc3d, resamples markers from old_rate to new_rate,
    then writes a TRC with a custom header format:

    1) PathFileType 4 (X/Y) TRC
    2) [DataRate] [CameraRate] [NumFrames] [NumMarkers] [Units] [OrigDataRate] [OrigDataStartFrame] [OrigNumFrames]
    3) #Frame   Time   Marker1   Marker2   ...
    4) [blank] [blank] X1  Y1  Z1   X2  Y2  Z2  ...
    5+) For each frame, one row: FrameIndex, TimeSec, M1_x, M1_y, M1_z, M2_x, ...

    All columns tab-separated.
    """
    # --- 1) Load C3D ---
    c3d = ezc3d.c3d(input_c3d_path)

    # Marker labels
    marker_labels = c3d["parameters"]["POINT"]["LABELS"]["value"]
    num_markers = len(marker_labels)

    # Marker data shape: (4, num_markers, num_frames)
    marker_data = c3d["data"]["points"]
    old_num_frames = marker_data.shape[2]

    # Original sampling rate
    if "RATE" in c3d["parameters"]["POINT"]:
        old_rate = c3d["parameters"]["POINT"]["RATE"]["value"][0]
    else:
        old_rate = c3d["header"]["frameRate"]

    # --- 2) Resample to new_rate ---
    new_marker_data, new_num_frames = resample_marker_data(marker_data, old_rate, new_rate)
    
    if assume_z_up:
        x_vals = new_marker_data[0, :, :]  # shape (num_markers, num_frames)
        y_vals = new_marker_data[1, :, :]
        z_vals = new_marker_data[2, :, :]

        # Apply rotation:
        #   X' =  X
        #   Y' =  Z
        #   Z' = -Y
        new_x = x_vals
        new_y = z_vals
        new_z = -y_vals

        new_marker_data[0, :, :] = new_x
        new_marker_data[1, :, :] = new_y
        new_marker_data[2, :, :] = new_z

    # Some metadata for the header
    data_rate = new_rate
    camera_rate = new_rate
    units = "mm"  # or "m" if your data is in meters
    orig_data_rate = old_rate
    orig_data_start_frame = 1
    orig_num_frames = old_num_frames

    # --- 3) Prepare header lines ---

    # Line 0
    line0 = "PathFileType\t4\t(X/Y/Z)\t" + f"{output_trc_path}"
    
    # Line 1
    line1 = "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames"

    # Line 2
    line2 = (
        f"{data_rate}\t{camera_rate}\t{new_num_frames}\t{num_markers}\t"
        f"{units}\t{orig_data_rate}\t{orig_data_start_frame}\t{orig_num_frames}"
    )

    # Line 3: #Frame   Time   Marker1   Marker2   ...
    header_line3 = "Frame#\tTime"
    for marker_name in marker_labels:
        header_line3 += f"\t\t{marker_name}"

    # Line 4: blank in first two columns, then X1  Y1  Z1  X2  Y2  Z2 ...
    # We build "X{m+1}, Y{m+1}, Z{m+1}" for each marker index.
    line4_parts = ["\t\t"]  # blank placeholders for the first two columns (#Frame, Time)
    for m in range(num_markers):
        line4_parts.append(f"X{m+1}")
        line4_parts.append(f"Y{m+1}")
        line4_parts.append(f"Z{m+1}")
    header_line4 = "\t".join(line4_parts)

    # --- 4) Prepare the data lines ---
    # Each frame: FrameIndex, TimeSec, M1_x, M1_y, M1_z, M2_x, ...
    data_lines = []
    dt = 1.0 / new_rate
    for f in range(new_num_frames):
        frame_index = f + 1
        time_sec = f * dt

        row_values = [str(frame_index), f"{time_sec:.5f}"]

        for m in range(num_markers):
            x = new_marker_data[0, m, f] 
            y = new_marker_data[1, m, f]
            z = new_marker_data[2, m, f]
            
            
            row_values.extend([f"{x:.5f}", f"{y:.5f}", f"{z:.5f}"])

        data_lines.append("\t".join(row_values))

    # --- 5) Write out the TRC file ---
    with open(output_trc_path, "w") as f_out:
        f_out.write(line0 + "\n")          # e.g. PathFileType 4 (X/Y) TRC
        f_out.write(line1 + "\n")          # e.g. PathFileType 4 (X/Y) TRC
        f_out.write(line2 + "\n")          # e.g. 100   100   101   5   mm   60   1   61
        f_out.write(header_line3 + "\n")   # e.g. #Frame Time Marker1 Marker2 ...
        f_out.write(header_line4 + "\n")   # e.g. <tab> <tab> X1 Y1 Z1   X2 Y2 Z2  ...
        f_out.write("\n")
        for line in data_lines:
            f_out.write(line + "\n")

    print(f"Resampled from {old_rate} Hz to {new_rate} Hz.")
    print(f"Custom TRC saved to: {os.path.abspath(output_trc_path)}")



#-----------------------------------------------------------------------------#
def c3d_to_trc_y_up_resample(
    input_c3d_path, 
    output_trc_path, 
    new_rate=None,
    assume_z_up=True
):
    """
    Convert a C3D to TRC, optionally rotating from Z-up to Y-up,
    and optionally resampling from old_rate to new_rate (Hz).

    Steps:
      1) Read the C3D with ezc3d.
      2) Extract marker data & labels.
      3) Rotate marker data if assume_z_up=True:
           X' = X
           Y' = Z
           Z' = -Y
         so that Y is up in the resulting TRC (OpenSim's convention).
      4) Resample if new_rate is specified (e.g., 100 Hz).
      5) Write the final marker data as TRC with correct header.

    Args:
      input_c3d_path (str): Path to the .c3d file
      output_trc_path (str): Path for the output .trc file
      new_rate (float or None): If given, resample to this rate (Hz).
                                If None, use the original rate from the C3D.
      assume_z_up (bool): Whether to apply the rotation from Z-up to Y-up.
                          If your data is already Y-up, set this False.

    Returns:
      None (writes out the TRC file).
    """
    # --- 1) Load the C3D file ---
    c3d = ezc3d.c3d(input_c3d_path)

    # Marker labels
    marker_labels = c3d["parameters"]["POINT"]["LABELS"]["value"]
    num_markers = len(marker_labels)

    # Marker data: shape (4, num_markers, num_frames)
    marker_data = c3d["data"]["points"]
    num_frames = marker_data.shape[2]

    # Original sampling rate
    if "RATE" in c3d["parameters"]["POINT"]:
        old_rate = c3d["parameters"]["POINT"]["RATE"]["value"][0]
    else:
        old_rate = c3d["header"]["frameRate"]

    # --- 2) Rotate from Z-up to Y-up (if specified) ---
    if assume_z_up:
        x_vals = marker_data[0, :, :]  # shape (num_markers, num_frames)
        y_vals = marker_data[1, :, :]
        z_vals = marker_data[2, :, :]

        # Apply rotation:
        #   X' =  X
        #   Y' =  Z
        #   Z' = -Y
        new_x = x_vals
        new_y = z_vals
        new_z = -y_vals

        marker_data[0, :, :] = new_x
        marker_data[1, :, :] = new_y
        marker_data[2, :, :] = new_z

    # --- 3) Resample if new_rate is given and different from old_rate ---
    if new_rate is not None and abs(new_rate - old_rate) > 1e-6:
        # Resample marker data
        marker_data, resampled_num_frames = resample_marker_data(marker_data, old_rate, new_rate)
        used_rate = new_rate
        final_num_frames = resampled_num_frames
    else:
        # No resampling
        used_rate = old_rate
        final_num_frames = num_frames

    # --- 4) Prepare TRC header info ---
    # For TRC: line1, line2, label line, then data lines
    # We'll assume units = "mm" unless you know your data is in meters
    units = "mm"
    data_rate = used_rate
    camera_rate = used_rate
    orig_data_rate = old_rate
    orig_data_start_frame = 1
    orig_num_frames = num_frames

    line1 = "PathFileType\t4\t(X/Y)\tTRC"
    line2 = (
        f"{data_rate}\t{camera_rate}\t{final_num_frames}\t{num_markers}\t"
        f"{units}\t{orig_data_rate}\t{orig_data_start_frame}\t{orig_num_frames}"
    )

    label_line = "Frame#\tTime"
    for label in marker_labels:
        label_line += f"\t{label}_X\t{label}_Y\t{label}_Z"

    # --- 5) Write the data lines ---
    dt = 1.0 / data_rate
    data_lines = []
    for f in range(final_num_frames):
        frame_idx = f + 1
        time_sec = f * dt
        row_values = [str(frame_idx), f"{time_sec:.5f}"]
        for m in range(num_markers):
            x = marker_data[0, m, f]
            y = marker_data[1, m, f]
            z = marker_data[2, m, f]
            row_values.extend([f"{x:.5f}", f"{y:.5f}", f"{z:.5f}"])
        data_lines.append("\t".join(row_values))

    # --- 6) Write out the TRC file ---
    with open(output_trc_path, "w") as f_out:
        f_out.write(line1 + "\n")
        f_out.write(line2 + "\n")
        f_out.write(label_line + "\n")
        for line in data_lines:
            f_out.write(line + "\n")

    print(f"Converted '{os.path.basename(input_c3d_path)}' -> '{os.path.basename(output_trc_path)}'")
    if assume_z_up:
        print("Applied Z-up -> Y-up rotation.")
    if new_rate is not None and abs(new_rate - old_rate) > 1e-6:
        print(f"Resampled from {old_rate} Hz to {new_rate} Hz.")
    else:
        print(f"Used original rate of {old_rate} Hz.")
    print("Done.")
    
    
#-----------------------------------------------------------------------------#
def read_trc_to_dataframe(trc_path):
    """
    Reads a TRC file with multiple header lines into a Pandas DataFrame.
    Returns the DataFrame.
    
    Assumes a format like:
      1) PathFileType line
      2) DataRate line
      3) #Frame   Time   Marker1   Marker2 ...
      4) [blank]  [blank]  X1  Y1  Z1   X2  Y2  Z2  ...
      5+) numeric data lines

    Adjust skip logic if your TRC differs.
    """
    with open(trc_path, 'r') as f:
        lines = [l.strip('\n') for l in f.readlines()]
    
    # 1. Find the line that starts with "#Frame" or "Frame#"
    data_start_idx = None
    for i, line in enumerate(lines):
        if line.startswith("#Frame") or line.startswith("Frame#"):
            data_start_idx = i
            break
    if data_start_idx is None:
        raise ValueError("Could not find a line starting with '#Frame' or 'Frame#' in the TRC file.")
    
    # 2. We expect the row with X1, Y1, Z1 to be right after that (data_start_idx+1).
    #    Then numeric data starts at (data_start_idx+2).
    header_line_1 = lines[data_start_idx].split()     # e.g. ['#Frame', 'Time', 'Marker1', 'Marker2', ...]
    header_line_2 = lines[data_start_idx+1].split()   # e.g. ['', '', 'X1', 'Y1', 'Z1', 'X2', 'Y2', 'Z2', ...]
    
    # 3. Combine header_line_1 and header_line_2 to form final column names.
    #    For example:
    #      #Frame  Time  Marker1  Marker2
    #      ""      ""    X1       Y1 ...
    #    => columns = [Frame, Time, Marker1_X, Marker1_Y, Marker1_Z, Marker2_X, ...]
    
    columns = []
    # The assumption: the first two entries are #Frame, Time; second row for those is blank.
    # After that, we pair up MarkerX name with X1/Y1/Z1 from header_line_2.
    
    # For safety, ensure both lines have the same length by padding if needed:
    max_len = max(len(header_line_1), len(header_line_2))
    col1 = (header_line_1 + [""]*(max_len - len(header_line_1)))[:max_len]
    col2 = (header_line_2 + [""]*(max_len - len(header_line_2)))[:max_len]
    
    for c1, c2 in zip(col1, col2):
        c1 = c1.strip()
        c2 = c2.strip()
        
        # If it's the #Frame or Frame#
        if c1.startswith("#Frame") or c1.startswith("Frame#"):
            columns.append("Frame")
        # If it's Time
        elif c1 == "Time":
            columns.append("Time")
        # If c1 is a marker name (like "Marker1") and c2 is dimension ("X1", "Y1", "Z1")
        elif c1 and c2 and c2[0] in ["X", "Y", "Z"]:
            # e.g. 'Marker1' and 'X1' => 'Marker1_X'
            marker_dim = c2[0]  # 'X' or 'Y' or 'Z'
            new_name = f"{c1}_{marker_dim}"
            columns.append(new_name)
        else:
            # Otherwise, maybe blank or unknown
            # We'll still append something to keep column count aligned
            columns.append(c1 if c1 else c2)
    
    # 4. Now read the numeric data using read_csv, skipping all lines up to data_start_idx+2
    numeric_start = data_start_idx + 2
    df = pd.read_csv(
        trc_path, 
        sep=r'\s+|\t+',   # split on whitespace or tabs
        engine='python', 
        skiprows=numeric_start, 
        header=None
    )
    
    # 5. Trim or pad columns if needed
    if len(df.columns) > len(columns):
        df = df.iloc[:, :len(columns)]
    elif len(df.columns) < len(columns):
        columns = columns[:len(df.columns)]
    
    df.columns = columns
    
    return df

#-----------------------------------------------------------------------------#
def read_trc_and_header_to_json(trc_file_path):
    """
    Reads a TRC file with a 4-line header and then numeric data.
    Returns:
      df: Pandas DataFrame of the numeric data
      header_json: JSON string containing parsed header info
    """
    # Read all lines
    with open(trc_file_path, 'r') as f:
        lines = [line.rstrip("\n") for line in f.readlines()]

    # --- 1) Extract the 4-line header ---
    # Adjust indices if your TRC has more or fewer header lines.
    line1 = lines[0]  # e.g. "PathFileType 4 (X/Y) TRC"
    line2 = lines[1]  # e.g. "100    100    101    5    mm    60    1    61"
    line3 = lines[2]  # e.g. "#Frame    Time    Marker1    Marker2"
    line4 = lines[3]  # e.g. "         X1      Y1      Z1      X2      Y2      Z2"

    # --- 2) Parse line2 into key-value pairs ---
    # Typical format: [DataRate, CameraRate, NumFrames, NumMarkers, Units, OrigDataRate, OrigDataStartFrame, OrigNumFrames]
    # We'll split by whitespace, convert to correct types, and label them:
    parts_line2 = line2.split()
    # Make sure we have at least 8 parts
    if len(parts_line2) < 8:
        raise ValueError("Line 2 of TRC header does not have enough fields (need 8).")

    data_rate = float(parts_line2[0])
    camera_rate = float(parts_line2[1])
    num_frames = int(parts_line2[2])
    num_markers = int(parts_line2[3])
    units = parts_line2[4]
    orig_data_rate = float(parts_line2[5])
    orig_data_start = int(parts_line2[6])
    orig_num_frames = int(parts_line2[7])

    # --- 3) Parse line3 & line4 for column labels ---
    # Example:
    # line3 => "#Frame    Time    Marker1    Marker2"
    # line4 => "         X1      Y1      Z1      X2      Y2      Z2"
    # The columns for numeric data typically combine these two lines.

    # Split them
    line3_parts = line3.split()
    line4_parts = line4.split()

    # For example:
    # line3_parts = ["#Frame", "Time", "Marker1", "Marker2"]
    # line4_parts = ["X1", "Y1", "Z1", "X2", "Y2", "Z2"]

    # We can store them directly in the JSON or build final column names used for the DataFrame.
    # Let's keep them in the JSON as raw strings:
    #   "header_line_3": "#Frame    Time    Marker1    Marker2"
    #   "header_line_4": "X1       Y1      Z1      X2      Y2      Z2"
    # Then, we combine them carefully to build DataFrame columns.

    # We'll store the original lines in the header JSON:
    header_dict = {
        "path_file_type_line": line1,
        "metadata_line": {
            "DataRate": data_rate,
            "CameraRate": camera_rate,
            "NumFrames": num_frames,
            "NumMarkers": num_markers,
            "Units": units,
            "OrigDataRate": orig_data_rate,
            "OrigDataStartFrame": orig_data_start,
            "OrigNumFrames": orig_num_frames
        },
        "header_line_3": line3,
        "header_line_4": line4
    }

    # Convert header_dict to JSON
    header_json = json.dumps(header_dict, indent=2)

    # --- 4) Build final column names for numeric data ---
    # Typically, the first two columns of numeric data = FrameIndex, Time
    # Then for each marker, we have (X, Y, Z) in the correct order.
    #
    # If line3_parts = ['#Frame','Time','Marker1','Marker2']
    # and line4_parts = ['X1','Y1','Z1','X2','Y2','Z2']
    # We want something like:
    # columns = ['Frame','Time','Marker1_X','Marker1_Y','Marker1_Z','Marker2_X','Marker2_Y','Marker2_Z']
    #
    # We can do a more robust approach by pairing line3 and line4 after skipping the first 2 items in line3.

    # Start columns with "Frame", "Time"
    final_columns = ["Frame", "Time"]
    # Then each marker name from line3_parts[2..] will pair with triplets from line4_parts
    marker_names = line3_parts[2:]  # e.g. ["Marker1","Marker2"]
    # line4_parts => ["X1","Y1","Z1","X2","Y2","Z2"]

    # We'll chunk line4_parts in groups of 3
    # For m in marker_names, we get 3 items from line4_parts
    if len(line4_parts) != 3 * len(marker_names):
        # Maybe there's spacing in line4 that doesn't match perfectly.
        # You might need a different approach if the lines contain blank columns.
        pass

    idx = 0
    for marker in marker_names:
        # e.g. marker = "Marker1", next 3 from line4_parts = ["X1","Y1","Z1"]
        if idx + 2 < len(line4_parts):
            xdim = line4_parts[idx]
            ydim = line4_parts[idx + 1]
            zdim = line4_parts[idx + 2]
            idx += 3
            # Build final col names: marker_X, marker_Y, marker_Z
            final_columns.append(f"{marker}_{xdim[0]}")  # e.g. "Marker1_X"
            final_columns.append(f"{marker}_{ydim[0]}")
            final_columns.append(f"{marker}_{zdim[0]}")
        else:
            break

    # --- 5) Read numeric data from lines[4 .. end] into DataFrame ---
    numeric_data_lines = lines[4:]  # from line 5 onward
    # We'll parse these lines with Pandas. They may be tab or whitespace separated.
    # We'll do a temporary approach: write them to a small buffer or parse them directly.
    
    from io import StringIO
    data_str = "\n".join(numeric_data_lines)
    df = pd.read_csv(StringIO(data_str), 
                     delim_whitespace=True,  # or sep="\t"
                     header=None)  # no header in the numeric rows themselves
    
    # If the file has consistent columns, df should have the right shape.
    # Now assign final_columns (or pad/truncate if needed).
    if len(df.columns) >= len(final_columns):
        df = df.iloc[:, :len(final_columns)]
    else:
        # If fewer columns than expected, truncate final_columns
        final_columns = final_columns[:len(df.columns)]
    
    df.columns = final_columns

    return df, header_json


#    df, header_json = read_trc_and_header_to_json(trc_path)



#-----------------------------------------------------------------------------#
def read_sto_and_header_to_json(sto_file_path):

    with open(sto_file_path, 'r') as f:
        lines = [line.rstrip("\n") for line in f]

    header_dict = {}
    data_start_idx = None
    col_header_line = None

    # 1. Parse lines until we reach 'endheader'
    for i, line in enumerate(lines):
        if line.strip().lower() == "endheader":
            # 'endheader' found; store index+1 as first data row (for column names)
            data_start_idx = i + 1
            break
        # If not 'endheader', attempt to parse key=value pairs
        if "=" in line:
            # e.g. "nRows=31"
            key_val = line.split("=", 1)  # split only on first '='
            if len(key_val) == 2:
                key, val = key_val[0].strip(), key_val[1].strip()
                header_dict[key] = val
        else:
            # If there's a line without '=', we can store it as well (or ignore)
            # e.g. "StorageFormatVersion=1" might come in, or something else
            # If you want to preserve every line exactly, store them separately
            pass

    if data_start_idx is None:
        raise ValueError("Could not find 'endheader' in the .sto file.")

    # 2. The next line after 'endheader' should be the column header line
    if data_start_idx < len(lines):
        col_header_line = lines[data_start_idx].strip()
        data_start_idx += 1
    else:
        raise ValueError("No column header line found after 'endheader'.")

    # 3. Parse the column names
    # Typically space or tab separated
    col_names = col_header_line.split()

    # 4. The rest of the lines are numeric data
    data_lines = lines[data_start_idx:]
    data_str = "\n".join(data_lines)

    # 5. Read into a Pandas DataFrame
    df = pd.read_csv(
        StringIO(data_str),
        delim_whitespace=True,  # or sep="\t" if strictly tab-delimited
        names=col_names,
        comment='#',  # Some STO files can have # comments. If not, remove this.
        header=None,
        engine='python'
    )

    # 6. Convert the header dict to JSON
    header_json = json.dumps(header_dict, indent=2)

    return df, header_json



#-----------------------------------------------------------------------------#
def read_c3d_to_df(c3d_path):
    """
    Reads marker data from a C3D file into a Pandas DataFrame using ezc3d.

    The resulting DataFrame will have columns like:
    ['Frame', 'Time', 'Marker1_X', 'Marker1_Y', 'Marker1_Z', 'Marker2_X', ...]
    for each frame.

    Args:
        c3d_path (str): Path to the C3D file.

    Returns:
        pd.DataFrame: DataFrame with marker data and time for each frame.
    """
    # 1) Load the C3D file with ezc3d
    c3d = ezc3d.c3d(c3d_path)

    # 2) Extract marker labels
    #    c3d["parameters"]["POINT"]["LABELS"]["value"] is a list of marker names
    marker_labels = c3d["parameters"]["POINT"]["LABELS"]["value"]
    n_markers = len(marker_labels)

    # 3) Extract the 3D marker data array
    #    shape: (4, nMarkers, nFrames)
    #    Indices: 0->X, 1->Y, 2->Z, 3->residual/error
    marker_data = c3d["data"]["points"]
    _, _, n_frames = marker_data.shape

    # 4) Determine the sampling rate (frames per second)
    #    Often stored in c3d["parameters"]["POINT"]["RATE"]["value"][0].
    #    If missing, you could fallback to c3d["header"]["frameRate"].
    if "RATE" in c3d["parameters"]["POINT"]:
        rate = c3d["parameters"]["POINT"]["RATE"]["value"][0]
    else:
        rate = c3d["header"]["frameRate"]
    # Time array for each frame: 0, 1/rate, 2/rate, ...
    times = [i / rate for i in range(n_frames)]

    # 5) Build a dictionary to construct the DataFrame.
    #    We'll include a Frame index (0..N-1) and Time.
    data_dict = {}
    data_dict["Frame"] = list(range(n_frames))
    data_dict["Time"] = times

    # For each marker, create 3 columns (X, Y, Z).
    for i, label in enumerate(marker_labels):
        x_values = marker_data[0, i, :]  # X over frames
        y_values = marker_data[1, i, :]  # Y over frames
        z_values = marker_data[2, i, :]  # Z over frames

        data_dict[f"{label}_X"] = x_values
        data_dict[f"{label}_Y"] = y_values
        data_dict[f"{label}_Z"] = z_values

    # 6) Convert the dictionary into a Pandas DataFrame
    df = pd.DataFrame(data_dict)

    return df    

if __name__ == "__main__":
     input_c3d = "DARI_DATA/Captury_Subj8_bilateral_squat02.c3d"
     output_trc = "DARI_DATA/Captury_Subj8_bilateral_squat02.trc"
     
     #c3d_to_trc_y_up_resample(input_c3d, output_trc, new_rate=100.0, assume_z_up=False)
     
     c3d_to_trc_custom(input_c3d, output_trc, 60, False)

