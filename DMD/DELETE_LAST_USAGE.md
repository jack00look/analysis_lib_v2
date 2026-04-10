# Delete Last Correction - User Guide

## Overview

The `--delete-last` flag allows you to **undo the last feedback correction** in the accumulation workflow. This is useful if:
- The last feedback made things worse instead of better
- You want to compare performance with/without the most recent correction
- You made a mistake with the kp or sigma parameters

## How It Works

When you run:
```bash
python auto_feedback_update.py --delete-last
```

The script will:

1. **Identify the last correction** in `feedback_sigmoids_list.py`
   - Shows you the filename and description of what will be deleted

2. **Ask for confirmation**
   - Type `yes` to proceed, anything else to cancel
   - Prevents accidental deletions

3. **Delete the correction file**
   - Removes the last sigmoid file from `magnetization_feedback/` folder
   - e.g., `sigmoid_center_interpolation_update_5.txt`

4. **Update the profile list**
   - Removes the entry from `feedback_sigmoids_list.py`
   - Now the list has one fewer correction

5. **Reload the profile**
   - Loads base DMD profile
   - Applies all *remaining* corrections (without the deleted one)
   - Automatically loads the updated profile to DMD server

## Example Usage

### Before deletion:
- 6 corrections applied (update_0 through update_5)
- Last one (update_5) made things worse

### Run delete command:
```bash
python auto_feedback_update.py --delete-last
```

### Output:
```
======================================================================
DELETE LAST CORRECTION
======================================================================

This will:
  1. Delete: sigmoid_center_interpolation_update_5.txt
     Description: Auto-added from waterfall
  2. Remove it from feedback_sigmoids_list.py
  3. Reload the profile with 5 corrections
  4. Load the updated profile to DMD server

Type 'yes' to confirm deletion: yes
```

### After deletion:
- File deleted from `magnetization_feedback/`
- Removed from `feedback_sigmoids_list.py`
- Profile recomputed with 5 corrections only
- New profile loaded to DMD server
- You can visualize and compare with `view_feedback_profiles.py`

## Important Notes

⚠️ **Warnings:**

1. **File Deletion is Permanent**
   - The correction file is permanently deleted (not recoverable)
   - You must confirm with 'yes' to proceed

2. **Profile is Immediately Reloaded to Server**
   - The DMD server is updated automatically
   - No manual server loading needed

3. **Cannot Delete If Only One Correction**
   - If feedback_sigmoids_list.py is empty, nothing will happen
   - At least one correction must exist

## Workflow Example

```bash
# Apply new feedback correction
python auto_feedback_update.py --kp 1.0 --sigma 2.0

# Check the result with visualization
python view_feedback_profiles.py

# If it made things worse, undo it
python auto_feedback_update.py --delete-last

# View the updated result
python view_feedback_profiles.py

# Now you can try again with different parameters
python auto_feedback_update.py --kp 0.5 --sigma 1.5
```

## Related Commands

```bash
# Normal workflow - add new correction
python auto_feedback_update.py --kp 0.8 --sigma 2.0

# Just visualize without changing anything
python view_feedback_profiles.py

# Manually edit the correction list
nano feedback_sigmoids_list.py

# Reset everything (delete ALL corrections)
python auto_feedback_update.py --reset

# Undo last correction
python auto_feedback_update.py --delete-last
```

## Troubleshooting

**Q: Nothing happens when I run --delete-last?**
- A: Make sure you have at least one correction in `feedback_sigmoids_list.py`
- A: Check that the corresponding file exists in `magnetization_feedback/`

**Q: I accidentally deleted the wrong correction!**
- A: Unfortunately, file deletion is permanent
- A: You could try to recover from git history if available
- A: Otherwise, you'll need to re-run feedback with the correction

**Q: How do I delete a specific correction (not the last one)?**
- A: Manually edit `feedback_sigmoids_list.py` and delete the entry
- A: Optionally delete the corresponding file from `magnetization_feedback/`
- A: Then reload with: `python auto_feedback_update.py --delete-last` (then re-add needed ones)
