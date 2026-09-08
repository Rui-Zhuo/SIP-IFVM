# SIP-IFVM code map and terminology

## Fixed terminology

| Term | Definition |
| --- | --- |
| **footpoint** | A magnetic-field-line endpoint or tracked position on the `r_index=0` inner spherical surface. |
| **crossing** | The intersection of that field line with the `R0=10 Rs` spherical surface. This is paired with a footpoint; it is not a footpoint. |
| **trace** | Integrate a magnetic field line within one simulation time. |
| **track** | Follow a footpoint or crossing through multiple simulation times. |
| **LCT** | Local-correlation tracking of `Br` magnetic elements on the `r_index=0` surface. |

`theta` is colatitude.  Longitude is `phi`; latitude is `90 deg - theta`.

## Packages and programs

| Location | Responsibility |
| --- | --- |
| `config.py` | Central paths, normalization constants, and directory configuration. |
| `utils.py` | Shared coordinate utilities and the differential-rotation law used for evolving analysis regions. |
| `reference/fortran_output_hdf5.txt` | Reference Fortran routines that write SIP solution and grid HDF5 files; retained as source-format documentation, not executed by Python. |
| `read_merged_sip_data.py` | Reads merged spherical HDF5 data and selects files by simulation time. |
| `merge_sip_grid.py` | Builds the regular global spherical grid from six SIP components. |
| `merge_sip_data.py` | Interpolates component solution data onto the global grid and converts to physical units. |
| `plot_sip_grid.py` | Visualizes the global merged grid. |
| `plot_sip_slices_scalar.py` | Draws scalar shell, equatorial, and meridional slices. |
| `plot_sip_slices_vector.py` | Draws vector-field slices. |
| `plot_sip_fieldlines.py` | Traces field lines and creates the interactive 3-D field-line visualization. The inner sphere uses the open/closed map plus the `Br=0` contour. |
| `classify_open_closed_states.py` | Traces field lines from multiple seed surfaces and writes the positive-open / closed / negative-open map on r_index=0. |
| `analyze_open_field_topology.py` | Calculates open-footpoint areas, fluxes, and internal boundary length in the configured coronal-hole ROI. |
| `track_crossings_connectivity.py` | Tracks model open field lines and writes paired inner footpoints and `R0` crossings. |
| `plot_tracked_crossings_connectivity_series.py` | Selects model-track boundary footpoints and plots their footpoint/crossing time series. |
| `plot_tracked_crossings_connectivity_map.py` | Maps selected model footpoints on the inner topology map or crossings on `Br(R0)`. |
| `calculate_lct_surface_velocity.py` | Estimates the r_index=0 horizontal magnetic-element velocity map by LCT. |
| `track_footpoints_lct.py` | Tracks all r_index=0 footpoints with LCT and writes persistent IDs. |
| `track_footpoints_lct_connectivity.py` | Re-traces every LCT footpoint at every time to obtain its `R0` crossing. |
| `plot_tracked_footpoints_lct_connectivity_series.py` | Plots selected all-ID LCT footpoint and traced-crossing coordinate series. |
| `plot_tracked_footpoints_lct_connectivity_map.py` | Maps LCT footpoints or their crossings. |
| `analyze_tracked_open_closed_states.py` | Samples the r_index=0 open/closed map at model- or LCT-tracked footpoints and saves state transitions. |
| `images_to_gif.py` | Converts image sequences to GIF. |
| `images_to_mp4.py` | Converts image sequences to MP4. |

## New output-name convention

Newly generated products use explicit surface roles:

| Product | Canonical name pattern |
| --- | --- |
| LCT velocity maps | `lct_footpoint_velocity.time.<t0>_to_<t1>.npz/png` |
| LCT footpoint tracks | `lct_footpoint_track.time.<t>.npz` |
| LCT all-ID trace connectivity | `lct_footpoint_connectivity.time.<t>.r.<R0>.npz` and manifest |
| Tracked open/closed state histories | `tracked_open_closed_states.crossings.npz` / `tracked_open_closed_states.lct.npz` |
| Selected model tracks | `selected_open_boundary_footpoints_crossings.r.<R0>.npz` |
| Model footpoint / crossing series | `open_boundary_footpoint_tracks.crossing-r.<R0>.png` / `open_boundary_crossing_tracks.r.<R0>.png` |
| Model footpoint / crossing maps | `open_boundary_footpoint_map.time.<t>.crossing-r.<R0>.png` / `open_boundary_crossing_map.time.<t>.r.<R0>.png` |
| Selected LCT tracks | `selected_lct_open_boundary_footpoints_crossings.r.<R0>.npz` |
| LCT footpoint / crossing series | `lct_open_boundary_footpoint_tracks.crossing-r.<R0>.png` / `lct_open_boundary_crossing_tracks.r.<R0>.png` |
| LCT footpoint / crossing maps | `lct_open_boundary_footpoint_map.time.<t>.r.<R0>.png` / `lct_open_boundary_crossing_map.time.<t>.r.<R0>.png` |
| Open-footpoint topology statistics | `open_field_footpoint_area_flux.npz/png`, `open_field_footpoint_boundary_length.png` |

The existing `open_closed_time.<t>.npz` is retained as the topology-map
interface because it is already the common input to downstream analysis.
Likewise, existing `track_open.*` files remain readable inputs.  In their NPZ
schema, legacy `r0_theta` and `r0_phi` mean the `R0` crossing coordinates.

## Recommended processing order

1. `merge_sip_grid.py` and `merge_sip_data.py`
2. `classify_open_closed_states.py`
3. Either model track: `track_crossings_connectivity.py`, or LCT track:
   `calculate_lct_surface_velocity.py` then `track_footpoints_lct.py`
4. Boundary selections and maps in the corresponding tracking scripts
5. `analyze_open_field_topology.py`
