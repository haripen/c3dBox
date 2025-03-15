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
def split_data_by_cycles(data_dict, cycle_periods, cycles_from_to):
    """
    Split the point and analog data into cycle segments based on cycle_periods.
    
    For each cycle (e.g. 'left_stride'), the function uses data_dict['point']['time'] and 
    data_dict['analog']['time'] to select the indices that fall within the cycle's time window.
    
    Additionally, if data_dict['events'] contains a 'kinetic' field, the kinetic flag for the 
    cycle is determined from the event closest to the cycle’s start time and added as a new key 
    'kinetic' in the cycle instance.
    
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
    
    # Retrieve event times and kinetic flags if available.
    if 'kinetic' in data_dict['events']:
        event_times = np.array(data_dict['events']['event_times'])
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
            
            # Set the kinetic flag based on the event closest to the cycle start time.
            if kinetic_flags is not None:
                idx = np.argmin(np.abs(event_times - start_time))
                cycle_instance['kinetic'] = kinetic_flags[idx]
            
            cycle_key = "cycle" + str(cycle_num)
            cycle_data[cycle_type][cycle_key] = cycle_instance
    
    return cycle_data