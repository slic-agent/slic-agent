"""Episode boundaries for trajectory-major, per-turn rollout rows."""


def episode_boundaries_from_lengths(lengths, traj_uids=None):
    if traj_uids is not None and len(traj_uids) != len(lengths):
        raise ValueError("Trajectory IDs and episode lengths must have the same row count.")
    boundaries = []
    offset = 0
    seen_uids = set()
    while offset < len(lengths):
        length = int(lengths[offset])
        if length < 1 or length != lengths[offset] or offset + length > len(lengths):
            raise ValueError(f"Invalid episode length {lengths[offset]} at row {offset}.")
        if any(value != length for value in lengths[offset:offset + length]):
            raise ValueError(f"Inconsistent repeated episode lengths at row {offset}.")
        if traj_uids is not None:
            uid = traj_uids[offset]
            if uid in seen_uids or any(value != uid for value in traj_uids[offset:offset + length]):
                raise ValueError(f"Episode length crosses trajectory boundaries at row {offset}.")
            seen_uids.add(uid)
        boundaries.append((offset, length))
        offset += length
    return boundaries
