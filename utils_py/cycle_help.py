import warnings
import numpy as np
def extract_cycle_periods(event_times, event_labels, cycles_from_to):
    """
    Extract cycle time ranges from aligned event times and labels.

    Supports cycle definitions with either 2 or 3 events.
    
    For a 2-event definition [start, end]:
      - If start and end labels are identical, consecutive occurrences are paired.
      - If they differ, the first occurrence of the end event following each start is used.
      
    For a 3-event definition [start, central, end]:
      - For each occurrence of the start event, the function searches for the first occurrence 
        of the central event and then the first occurrence of the end event following the central event.
      - If the central event is not found after a start event, a warning is issued and that cycle instance is skipped.
      - If the end event is not found after the central event, the cycle is considered incomplete and skipped silently.
    
    Parameters:
        event_times (list or array): The times corresponding to each event.
        event_labels (list): The event labels, aligned with event_times.
        cycles_from_to (dict): A dictionary mapping cycle names to a list of event labels.
                               The list must have either 2 or 3 elements.
    
    Returns:
        dict: A dictionary where each key is a cycle name and the value is another dictionary.
              In the inner dictionary, each key is the cycle number (starting at 1) and the value
              is a tuple defining that cycle's time range.
              - For 2-event cycles, the tuple is (start_time, end_time).
              - For 3-event cycles, the tuple is (start_time, central_time, end_time).
    """
    cycles = {}
    for cycle_name, events_seq in cycles_from_to.items():
        cycles[cycle_name] = {}
        if len(events_seq) == 2:
            start_label, end_label = events_seq
            if start_label == end_label:
                indices = [i for i, lab in enumerate(event_labels) if lab == start_label]
                for j in range(len(indices) - 1):
                    cycles[cycle_name][j + 1] = (event_times[indices[j]], event_times[indices[j + 1]])
            else:
                start_indices = [i for i, lab in enumerate(event_labels) if lab == start_label]
                count = 1
                for start_index in start_indices:
                    for i in range(start_index + 1, len(event_labels)):
                        if event_labels[i] == end_label:
                            cycles[cycle_name][count] = (event_times[start_index], event_times[i])
                            count += 1
                            break
        elif len(events_seq) == 3:
            start_label, central_label, end_label = events_seq
            start_indices = [i for i, lab in enumerate(event_labels) if lab == start_label]
            count = 1
            for start_index in start_indices:
                # Find the first central event after the start event.
                central_index = None
                for j in range(start_index + 1, len(event_labels)):
                    if event_labels[j] == central_label:
                        central_index = j
                        break
                if central_index is None:
                    warnings.warn(
                        f"Event sequence corrupted for cycle '{cycle_name}' at start index {start_index}: "
                        f"missing central event '{central_label}'. Skipping this cycle instance."
                    )
                    continue  # Skip this cycle instance.

                # Find the first end event after the central event.
                end_index = None
                for k in range(central_index + 1, len(event_labels)):
                    if event_labels[k] == end_label:
                        end_index = k
                        break
                if end_index is None:
                    # Incomplete cycle instance (not defined), so skip it silently.
                    continue

                cycles[cycle_name][count] = (event_times[start_index],
                                             event_times[central_index],
                                             event_times[end_index])
                count += 1
        else:
            warnings.warn(f"Cycle definition for '{cycle_name}' must have 2 or 3 events, got {len(events_seq)}. Skipping.")
    return cycles

# --- Split point and analog data by cycles ---
def split_data_by_cycles(data_dict, cycle_periods, cycles_from_to, add_contralateral=False):
    """
    Split the point and analog data into cycle segments based on cycle_periods.
    
    For each cycle (e.g. 'left_stride'), the function uses data_dict['point']['time'] and 
    data_dict['analog']['time'] to select the indices that fall within the cycle's time window.
    
    Additionally, if data_dict['events'] contains a 'kinetic' field, the kinetic flag for the 
    cycle is determined from the event closest to the cycle’s start time and added as a new key 
    'kinetic' in the cycle instance.
    
    Optionally (when add_contralateral=True), for gait locomotion cycles (left_stride/right_stride)
    the function also adds the contralateral *_Foot_Off and *_Foot_Strike events with time, frame,
    and percent (or NaN if not found).
    
    Returns:
        dict: A nested dictionary structured as:
              { cycle_type: { "cycle1": { 'point': { ... },
                                          'analog': { ... },
                                          'kinetic': True/False,
                                          [central_event_label]: { ... } (if applicable)
                                        },
                                "cycle2": { ... },
                                ...
                              }
              }
    """
    cycle_data = {}
    
    # Global arrays for point and analog times and frames.
    global_pt_time = np.array(data_dict['point']['time'])
    global_pt_frames = np.array(data_dict['point']['frames'])
    global_an_time = np.array(data_dict['analog']['time'])
    
    # Events used for contralateral Foot Off lookup (and kinetic if present)
    ev_times  = np.array(data_dict['events']['event_times'], dtype=float) if 'events' in data_dict else np.array([])
    ev_labels = list(data_dict['events']['event_labels']) if 'events' in data_dict else []
    
    # Retrieve event times and kinetic flags if available.
    if 'events' in data_dict and 'kinetic' in data_dict['events']:
        kinetic_flags = data_dict['events']['kinetic']
    else:
        kinetic_flags = None

    for cycle_type, cycles in cycle_periods.items():
        cycle_data[cycle_type] = {}
        # Determine if a central event is defined (3-event cycle) for this cycle type.
        has_central = (len(cycles_from_to.get(cycle_type, [])) == 3)
        if has_central:
            central_label = cycles_from_to[cycle_type][1].replace(' ', '_')
        
        for cycle_num, period in cycles.items():
            # For 2-event cycles, period is (start, end)
            # For 3-event cycles, period is (start, central, end)
            start_time = period[0]
            end_time = period[-1]
            
            # Determine indices for point and analog data within the cycle window.
            pt_idx = np.where((global_pt_time >= start_time) & (global_pt_time <= end_time))[0]
            an_idx = np.where((global_an_time >= start_time) & (global_an_time <= end_time))[0]
            
            # Slice point data.
            point_cycle = {}
            for key, arr in data_dict['point'].items():
                arr_np = np.array(arr)
                point_cycle[key] = arr_np[pt_idx] if pt_idx.size > 0 else arr_np[0:0]
            
            # Slice analog data.
            analog_cycle = {}
            for key, arr in data_dict['analog'].items():
                arr_np = np.array(arr)
                analog_cycle[key] = arr_np[an_idx] if an_idx.size > 0 else arr_np[0:0]
            
            cycle_instance = {'point': point_cycle, 'analog': analog_cycle}
            
            # If there is a central event (3-event cycle), add its details.
            if has_central:
                central_time = period[1]
                pt_central_idx = np.argmin(np.abs(global_pt_time - central_time))
                central_frame = global_pt_frames[pt_central_idx]
                cycle_duration = end_time - start_time
                cycle_percentage = 0.0
                if cycle_duration != 0:
                    cycle_percentage = ((central_time - start_time) / cycle_duration) * 100
                cycle_instance[central_label] = {
                    'time': central_time,
                    'frame': central_frame,
                    'percent': cycle_percentage
                }
                
            # --- Add contralateral *_Foot_Off for locomotion (if requested) ---
            # Only for typical gait cycles that have a central event: left/right_stride.
            if add_contralateral and has_central and cycle_type in ('left_stride', 'right_stride'):
                # Expected contralateral key to add into the cycle dict
                expected_key = 'Right_Foot_Off' if cycle_type == 'left_stride' else 'Left_Foot_Off'

                # Guard in case events are missing
                if ev_times.size == 0 or not ev_labels:
                    cycle_instance[expected_key] = {'time': np.nan, 'frame': np.nan, 'percent': np.nan}
                else:
                    # Central label in "spaced" form to match ev_labels (e.g., "Left Foot Off")
                    central_text = cycles_from_to[cycle_type][1]  # e.g. "Left Foot Off"
                    central_time = period[1]

                    # 1) Find the index in ev_labels that corresponds to THIS cycle's central Foot Off,
                    #    preferring the one whose timestamp is closest to period[1].
                    cand_idx = [i for i, lab in enumerate(ev_labels) if lab == central_text]
                    if cand_idx:
                        diffs = [abs(ev_times[i] - central_time) for i in cand_idx]
                        j = cand_idx[int(np.argmin(diffs))]
                    else:
                        # Fallback: nearest "* Foot Off" to central_time if exact label match is absent
                        off_idx = [i for i, lab in enumerate(ev_labels) if 'Foot Off' in lab]
                        if off_idx:
                            diffs = [abs(ev_times[i] - central_time) for i in off_idx]
                            j = off_idx[int(np.argmin(diffs))]
                        else:
                            j = None

                    # 2) Walk backward to find the immediate previous "* Foot Off" (the contralateral one in locomotion).
                    contra_time = np.nan
                    contra_ok = False
                    if j is not None:
                        k = None
                        for idx in range(j - 1, -1, -1):
                            if 'Foot Off' in ev_labels[idx]:
                                k = idx
                                break
                        if k is not None:
                            # We found the previous Foot Off; ensure it is the contralateral side we expect.
                            found_key = ev_labels[k].replace(' ', '_')  # "Right_Foot_Off" or "Left_Foot_Off"
                            if found_key == expected_key:
                                contra_time = float(ev_times[k])
                                contra_ok = True

                    # 3) Compute frame and percent if found and inside the cycle (good data should be inside).
                    if contra_ok and not np.isnan(contra_time):
                        pt_contra_idx = int(np.argmin(np.abs(global_pt_time - contra_time)))
                        contra_frame = float(global_pt_frames[pt_contra_idx])
                        cycle_duration = end_time - start_time
                        contra_percent = np.nan
                        if cycle_duration != 0:
                            contra_percent = ((contra_time - start_time) / cycle_duration) * 100.0

                        cycle_instance[expected_key] = {
                            'time': contra_time,
                            'frame': contra_frame,
                            'percent': contra_percent
                        }
                    else:
                        # Not found or not contralateral -> fill with NaNs per requirement
                        cycle_instance[expected_key] = {'time': np.nan, 'frame': np.nan, 'percent': np.nan}
            # --- end contralateral block ---
            # --- Add contralateral *_Foot_Strike for locomotion (if requested) ---
            if add_contralateral and cycle_type in ('left_stride', 'right_stride'):
                expected_strike_key = 'Right_Foot_Strike' if cycle_type == 'left_stride' else 'Left_Foot_Strike'

                # Guard if events are not present
                if ev_times.size == 0 or not ev_labels:
                    cycle_instance[expected_strike_key] = {'time': np.nan, 'frame': np.nan, 'percent': np.nan}
                else:
                    # Start event (e.g., "Left Foot Strike" for left_stride)
                    start_text = cycles_from_to[cycle_type][0]   # e.g., "Left Foot Strike" or "Right Foot Strike"
                    start_t = start_time

                    # 1) Find the index of THIS cycle's start strike (nearest in time with matching label)
                    cand_start_idx = [i for i, lab in enumerate(ev_labels) if lab == start_text]
                    if cand_start_idx:
                        diffs = [abs(ev_times[i] - start_t) for i in cand_start_idx]
                        j0 = cand_start_idx[int(np.argmin(diffs))]
                    else:
                        # Fallback: nearest generic Foot Strike to start_t
                        strike_idx = [i for i, lab in enumerate(ev_labels) if 'Foot Strike' in lab]
                        if strike_idx:
                            diffs = [abs(ev_times[i] - start_t) for i in strike_idx]
                            j0 = strike_idx[int(np.argmin(diffs))]
                        else:
                            j0 = None

                    # 2) The contralateral strike in normal locomotion is the NEXT "* Foot Strike" after the start strike
                    contra_time = np.nan
                    contra_ok = False
                    if j0 is not None:
                        k = None
                        for idx in range(j0 + 1, len(ev_labels)):
                            if 'Foot Strike' in ev_labels[idx]:
                                k = idx
                                break
                        if k is not None:
                            found_key = ev_labels[k].replace(' ', '_')   # "Right_Foot_Strike" or "Left_Foot_Strike"
                            if found_key == expected_strike_key:
                                contra_time = float(ev_times[k])
                                # Optional: ensure it lies inside this cycle window [start_time, end_time]
                                if (contra_time >= start_time) and (contra_time <= end_time):
                                    contra_ok = True

                    # 3) Compute frame and percent or fill NaNs
                    if contra_ok and not np.isnan(contra_time):
                        pt_contra_idx = int(np.argmin(np.abs(global_pt_time - contra_time)))
                        contra_frame = float(global_pt_frames[pt_contra_idx])
                        cycle_duration = end_time - start_time
                        contra_percent = ((contra_time - start_time) / cycle_duration) * 100.0 if cycle_duration != 0 else np.nan

                        cycle_instance[expected_strike_key] = {
                            'time': contra_time,
                            'frame': contra_frame,
                            'percent': contra_percent
                        }
                    else:
                        cycle_instance[expected_strike_key] = {'time': np.nan, 'frame': np.nan, 'percent': np.nan}
            # --- end contralateral strike block ---

            # Set the kinetic flag based on the event closest to the cycle start time.
            if kinetic_flags is not None and ev_times.size:
                idx = int(np.argmin(np.abs(ev_times - start_time)))
                # Guard in case of any unexpected length mismatch
                if idx < len(kinetic_flags):
                    cycle_instance['kinetic'] = kinetic_flags[idx]
                else:
                    cycle_instance['kinetic'] = bool(kinetic_flags[-1])  # fallback

            cycle_key = "cycle" + str(cycle_num)
            cycle_data[cycle_type][cycle_key] = cycle_instance
            
    return cycle_data