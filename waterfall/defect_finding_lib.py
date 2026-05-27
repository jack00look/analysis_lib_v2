import numpy as np
from scipy.ndimage import gaussian_filter1d


DEFAULT_DEFECT_CONFIG = {
	'gaussian_sigma': 0.3,
	'derivative_threshold': 0.0,
	'derivative_threshold_pos': 0.1,
	'derivative_threshold_neg': -0.1,
	'peak_step_filter_enabled': False,
	'peak_step_filter_window_px': 4,
	'peak_step_filter_min_abs_delta_m': 0.7,
	'min_same_sign_peak_distance_um': 7.0,
	'zero_crossing_min_distance_um': 3.0,
}


def normalize_defect_config(defect_cfg=None):
	cfg = dict(DEFAULT_DEFECT_CONFIG)
	if isinstance(defect_cfg, dict):
		cfg.update(defect_cfg)

	cfg['gaussian_sigma'] = float(cfg.get('gaussian_sigma', 0.3))
	base_thr = float(cfg.get('derivative_threshold', 0.0))
	cfg['derivative_threshold_pos'] = float(cfg.get('derivative_threshold_pos', base_thr))
	cfg['derivative_threshold_neg'] = float(cfg.get('derivative_threshold_neg', -abs(cfg['derivative_threshold_pos'])))
	cfg['peak_step_filter_enabled'] = bool(cfg.get('peak_step_filter_enabled', True))
	cfg['peak_step_filter_window_px'] = max(1, int(cfg.get('peak_step_filter_window_px', 4)))
	cfg['peak_step_filter_min_abs_delta_m'] = max(0.0, float(cfg.get('peak_step_filter_min_abs_delta_m', 0.7)))
	cfg['min_same_sign_peak_distance_um'] = max(0.0, float(cfg.get('min_same_sign_peak_distance_um', 7.0)))
	cfg['zero_crossing_min_distance_um'] = max(0.0, float(cfg.get('zero_crossing_min_distance_um', 3.0)))
	return cfg


def _count_peak_indices_pos(d_signal, threshold):
	if d_signal.size < 3:
		return np.array([], dtype=int)
	local_max = (d_signal[1:-1] >= d_signal[:-2]) & (d_signal[1:-1] > d_signal[2:])
	above = d_signal[1:-1] >= threshold
	return np.where(local_max & above)[0] + 1


def _count_peak_indices_neg(d_signal, threshold):
	if d_signal.size < 3:
		return np.array([], dtype=int)
	local_min = (d_signal[1:-1] <= d_signal[:-2]) & (d_signal[1:-1] < d_signal[2:])
	below = d_signal[1:-1] <= threshold
	return np.where(local_min & below)[0] + 1


def _merge_close_same_sign_peaks(peak_idx, x_axis_um, min_sep_um, opposite_peaks=None):
	"""
	Merge peaks of the same sign if they are closer than min_sep_um.
	Only merge if there is NO opposite-sign peak between them.
	
	Parameters:
	-----------
	peak_idx : array-like
		Indices of peaks to potentially merge
	x_axis_um : array-like
		Position axis in micrometers
	min_sep_um : float
		Minimum separation for merging
	opposite_peaks : array-like, optional
		Positions (in um) of opposite-sign peaks. Peaks separated by opposite-sign peaks won't merge.
	
	Returns:
	--------
	merged_idx : ndarray
		Merged peak indices
	merged_x : ndarray
		Merged peak positions in um
	"""
	idx = np.asarray(peak_idx, dtype=int)
	x_axis_um = np.asarray(x_axis_um, dtype=float)
	if idx.size == 0:
		return np.array([], dtype=int), np.array([], dtype=float)

	idx = np.unique(np.sort(idx))
	if float(min_sep_um) <= 0:
		return idx, x_axis_um[idx]
	
	# Convert opposite peaks to array if provided
	if opposite_peaks is not None:
		opposite_peaks = np.asarray(opposite_peaks, dtype=float)
	else:
		opposite_peaks = np.array([], dtype=float)

	merged_idx = []
	merged_x = []

	cluster = [int(idx[0])]
	for ii in idx[1:]:
		ii = int(ii)
		x_ii = float(x_axis_um[ii])
		x_prev = float(x_axis_um[cluster[-1]])
		
		# Check if close enough to potentially merge
		if abs(x_ii - x_prev) < float(min_sep_um):
			# Check if there's an opposite-sign peak between them
			x_start = min(x_prev, x_ii)
			x_end = max(x_prev, x_ii)
			has_opposite_between = np.any((opposite_peaks > x_start) & (opposite_peaks < x_end))
			
			if not has_opposite_between:
				# Safe to merge
				cluster.append(ii)
				continue

		# Not close or blocked by opposite peak - finalize current cluster
		if len(cluster) == 1:
			i0 = int(cluster[0])
			merged_idx.append(i0)
			merged_x.append(float(x_axis_um[i0]))
		else:
			i0 = int(cluster[0])
			i1 = int(cluster[-1])
			xmid = 0.5 * (float(x_axis_um[i0]) + float(x_axis_um[i1]))
			imid = int(np.argmin(np.abs(x_axis_um - xmid)))
			merged_idx.append(imid)
			merged_x.append(float(xmid))
		cluster = [ii]

	# Finalize last cluster
	if len(cluster) == 1:
		i0 = int(cluster[0])
		merged_idx.append(i0)
		merged_x.append(float(x_axis_um[i0]))
	else:
		i0 = int(cluster[0])
		i1 = int(cluster[-1])
		xmid = 0.5 * (float(x_axis_um[i0]) + float(x_axis_um[i1]))
		imid = int(np.argmin(np.abs(x_axis_um - xmid)))
		merged_idx.append(imid)
		merged_x.append(float(xmid))

	return np.asarray(merged_idx, dtype=int), np.asarray(merged_x, dtype=float)


def _find_opposite_peak_x(d_roi, x_roi_um, sign_run, i_start, i_end):
	if d_roi.size < 3:
		return None
	i0 = int(max(1, min(i_start, i_end)))
	i1 = int(min(d_roi.size - 2, max(i_start, i_end)))
	if i1 < i0:
		return None

	seg = d_roi[i0:i1 + 1]
	if seg.size == 0:
		return None

	if sign_run == 'pos':
		rel = int(np.argmin(seg))
		idx = i0 + rel
		is_local = bool((d_roi[idx] <= d_roi[idx - 1]) and (d_roi[idx] < d_roi[idx + 1]))
		passes = bool(d_roi[idx] < 0.0)
	else:
		rel = int(np.argmax(seg))
		idx = i0 + rel
		is_local = bool((d_roi[idx] >= d_roi[idx - 1]) and (d_roi[idx] > d_roi[idx + 1]))
		passes = bool(d_roi[idx] > 0.0)

	if not (is_local and passes):
		return None
	return float(x_roi_um[idx])


def _passes_peak_step_filter(m_roi, peak_idx, window_px, min_abs_delta):
	i = int(peak_idx)
	l0 = i - int(window_px)
	l1 = i
	r0 = i + 1
	r1 = i + 1 + int(window_px)
	if l0 < 0 or r1 > m_roi.size:
		return False
	left_avg = float(np.mean(m_roi[l0:l1]))
	right_avg = float(np.mean(m_roi[r0:r1]))
	return abs(right_avg - left_avg) >= float(min_abs_delta)


def find_defects_in_profile(m_profile, x_axis_um, defect_cfg, x_min_um, x_max_um):
	cfg = normalize_defect_config(defect_cfg)

	x_axis_um = np.asarray(x_axis_um, dtype=float)
	m_sig = np.asarray(m_profile, dtype=float)
	if m_sig.size != x_axis_um.size or m_sig.size < 3:
		return {
			'threshold_x_all': np.array([], dtype=float),
			'threshold_x_pos': np.array([], dtype=float),
			'threshold_x_neg': np.array([], dtype=float),
			'corrected_x': np.array([], dtype=float),
			'rejected_x': np.array([], dtype=float),
			'd_roi': np.array([], dtype=float),
			'm_roi': np.array([], dtype=float),
			'x_roi_um': np.array([], dtype=float),
			'count': 0,
		}

	if cfg['gaussian_sigma'] > 0:
		m_sig = gaussian_filter1d(m_sig, sigma=cfg['gaussian_sigma'])

	d_sig = np.gradient(m_sig)

	xmin_ind = int(np.argmin(np.abs(x_axis_um - float(x_min_um))))
	xmax_ind = int(np.argmin(np.abs(x_axis_um - float(x_max_um))))
	if xmax_ind < xmin_ind:
		xmin_ind, xmax_ind = xmax_ind, xmin_ind

	if d_sig.size < 3 or xmax_ind - xmin_ind < 2:
		return {
			'threshold_x_all': np.array([], dtype=float),
			'threshold_x_pos': np.array([], dtype=float),
			'threshold_x_neg': np.array([], dtype=float),
			'corrected_x': np.array([], dtype=float),
			'rejected_x': np.array([], dtype=float),
			'd_roi': np.array([], dtype=float),
			'm_roi': np.array([], dtype=float),
			'x_roi_um': np.array([], dtype=float),
			'count': 0,
		}

	d_roi = d_sig[xmin_ind:xmax_ind + 1]
	m_roi = m_sig[xmin_ind:xmax_ind + 1]
	x_roi_um = x_axis_um[xmin_ind:xmax_ind + 1]

	peak_pos_raw = _count_peak_indices_pos(d_roi, cfg['derivative_threshold_pos'])
	peak_neg_raw = _count_peak_indices_neg(d_roi, cfg['derivative_threshold_neg'])

	peak_pos_roi = []
	peak_neg_roi = []
	rejected_idx = []

	for idxp in peak_pos_raw:
		if cfg['peak_step_filter_enabled'] and not _passes_peak_step_filter(
			m_roi,
			idxp,
			cfg['peak_step_filter_window_px'],
			cfg['peak_step_filter_min_abs_delta_m'],
		):
			rejected_idx.append(int(idxp))
		else:
			peak_pos_roi.append(int(idxp))

	for idxn in peak_neg_raw:
		if cfg['peak_step_filter_enabled'] and not _passes_peak_step_filter(
			m_roi,
			idxn,
			cfg['peak_step_filter_window_px'],
			cfg['peak_step_filter_min_abs_delta_m'],
		):
			rejected_idx.append(int(idxn))
		else:
			peak_neg_roi.append(int(idxn))

	peak_pos_roi, peak_pos_x = _merge_close_same_sign_peaks(
		peak_pos_roi,
		x_roi_um,
		cfg['min_same_sign_peak_distance_um'],
		opposite_peaks=x_roi_um[peak_neg_roi] if len(peak_neg_roi) > 0 else None,
	)
	peak_neg_roi, peak_neg_x = _merge_close_same_sign_peaks(
		peak_neg_roi,
		x_roi_um,
		cfg['min_same_sign_peak_distance_um'],
		opposite_peaks=x_roi_um[peak_pos_roi] if len(peak_pos_roi) > 0 else None,
	)

	peak_thr_x = np.concatenate([peak_pos_x, peak_neg_x]).astype(float)
	events = []
	for idxp in peak_pos_roi:
		events.append((int(idxp), 'pos'))
	for idxn in peak_neg_roi:
		events.append((int(idxn), 'neg'))
	events.sort(key=lambda e: e[0])

	corrected_x = []
	j = 0
	while j < len(events):
		sign_j = events[j][1]
		k = j
		while k < len(events) and events[k][1] == sign_j:
			k += 1
		run = events[j:k]

		if len(run) >= 2:
			candidates = []
			for p in range(len(run) - 1):
				i_start = int(run[p][0])
				i_end = int(run[p + 1][0])
				sign_pair = run[p][1]
				ox = _find_opposite_peak_x(d_roi, x_roi_um, sign_pair, i_start, i_end)
				if ox is not None:
					candidates.append(float(ox))

			for cx in candidates:
				if peak_thr_x.size == 0:
					corrected_x.append(float(cx))
					continue
				min_dist = float(np.min(np.abs(peak_thr_x - float(cx))))
				if min_dist >= cfg['zero_crossing_min_distance_um']:
					corrected_x.append(float(cx))
		j = k

	if len(corrected_x) > 1:
		corrected_x = list(np.unique(np.round(np.asarray(corrected_x, dtype=float), 6)))

	threshold_x_all = np.concatenate([peak_pos_x, peak_neg_x]).astype(float)
	rejected_x = np.asarray([float(x_roi_um[int(idxr)]) for idxr in rejected_idx], dtype=float)
	corrected_x = np.asarray(corrected_x, dtype=float)

	return {
		'threshold_x_all': threshold_x_all,
		'threshold_x_pos': np.asarray(peak_pos_x, dtype=float),
		'threshold_x_neg': np.asarray(peak_neg_x, dtype=float),
		'corrected_x': corrected_x,
		'rejected_x': rejected_x,
		'd_roi': np.asarray(d_roi, dtype=float),
		'm_roi': np.asarray(m_roi, dtype=float),
		'x_roi_um': np.asarray(x_roi_um, dtype=float),
		'count': int(len(threshold_x_all) + len(corrected_x)),
	}
