path = r'D:\3_document\4_research\NLOS Signal Identification and Correction\model\part3_ResidualFeedbackAndOnline_Correction\model\run_module3.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Change 1: Add flags
old1 = 'def load_mog_cache(dataset_name):'
new1 = '# v4: Disable posterior correction (harmful per ablation) and TCN (zero marginal effect)\nUSE_POSTERIOR_CORRECTION = False\nUSE_TCN = False\n\ndef load_mog_cache(dataset_name):'
content = content.replace(old1, new1, 1)
print('1: flags added')

# Change 2: Posterior init
content = content.replace('    posterior_corrector = PosteriorPlosCorrector()', '    posterior_corrector = PosteriorPlosCorrector() if USE_POSTERIOR_CORRECTION else None', 1)
print('2: posterior init guarded')

# Change 3: TCN solver
content = content.replace('    fg_tcn_solver = make_fg_tcn_solver(dataset_name)', '    fg_tcn_solver = make_fg_tcn_solver(dataset_name) if USE_TCN else None', 1)
print('3: TCN solver guarded')

# Change 4: Apply correction
old4 = '        mog_corrected = posterior_corrector.apply_correction(mog)'
new4 = '        if posterior_corrector is not None:\n            mog_corrected = posterior_corrector.apply_correction(mog)\n        else:\n            mog_corrected = mog'
content = content.replace(old4, new4, 1)
print('4: apply_correction guarded')

# Change 5: Update from residuals
old5 = '        posterior_corrector.update_from_residuals(obs_list, mog, pos_adaptive, sv_positions)'
new5 = '        if posterior_corrector is not None:\n            posterior_corrector.update_from_residuals(obs_list, mog, pos_adaptive, sv_positions)'
content = content.replace(old5, new5, 1)
print('5: update_from_residuals guarded')

# Change 6: Diagnostics
old6 = "    report['posterior_correction'] = posterior_corrector.get_diagnostics()"
new6 = "    if posterior_corrector is not None:\n        report['posterior_correction'] = posterior_corrector.get_diagnostics()\n    else:\n        report['posterior_correction'] = {'status': 'disabled (v4)'}"
content = content.replace(old6, new6, 1)
print('6: diagnostics guarded')

# Change 7: Version
content = content.replace("print('Module 3 v2: Residual Feedback + TCN + Per-Dataset Tuning')", "print('Module 3 v4: Adaptive Selection (no posterior, no TCN)')", 1)
print('7: version updated')

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print('All v4 changes applied!')
