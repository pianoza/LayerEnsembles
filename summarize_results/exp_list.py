def get_experiments_list(dataset):
	if dataset == 'inbreast':
		experiments = {
			'Default': 'Default'
		}
	elif dataset == 'bcdr':
		experiments = {
			# 'Plain': 'BCDR-Plain-Aug', # <- This is in the paper # Plain-Aug <-changed this name for plots
			# 'DE': 'DE-BCDR-GoldStandard-Aug',  # <- this is in the paper # DE-Aug <-changed this name for plots
			# 'LE': 'BCDR-TestRun-3-T7', # <- This is in the paper
			# 'Plain': 'BCDR-Plain',
			# 'DE': 'DE-BCDR-GoldStandard',
			# 'LE': 'BCDR-TestRun-2',  # resnet18
			# 'LE-STAPLE': 'BCDR-TestRun-2-STAPLE',  # resnet18 with STAPLE
			# 'LE-Aug': 'BCDR-TestRun-3',  # resnet18 with Aug
			# 'LE-Aug-STAPLE': 'BCDR-TestRun-3-STAPLE',  # resnet18 with Aug and STAPLE
			# 'LE-Aug-last': 'BCDR-TestRun-3-LastLayerOutput',  # resnet18 with Aug and last layer output
			# 'LE': 'BCDR-TestRun-3-STAPLE-T10', # (LE-Aug-STAPLE) resnet18 with Aug and STAPLE, T=10, uncertainty maps are from T10 but AULA is from T7, this gives a better calibration!
			# 'LE-T10': 'BCDR-TestRun-3-T10', 
			# 'LE-Aug-STAPLE-lAULA': 'BCDR-TestRun-3-STAPLE-lAULA', # resnet18 with Aug and STAPLE, Layer agreement with the last layer output not the adjacecnt one
			'0%': 'BCDR-TestRun-3-T10-PD-0pCorr',
			'50%': 'BCDR-TestRun-3-T10-PD-50pCorr-alt',
			'100%': 'BCDR-TestRun-3-T10-PD-100pCorr-alt',
		}
	elif dataset == 'mnm':
		experiments = {
			'Plain': 'MnM-Plain-CE',  # Plain-CE <-changed this name for plots
			'DE': 'MnM-DE-GoldStandard-CE',  # Cross entropy, with Aug; DE-CE <-changed this name for plots
			# 'LE': 'MnM-TestRun-4-T5-inPaper',  #  T5, resnet18, with Aug, 200 epochs, STAPLE; LE-CE-T5 <-changed this name for plots
			'LE': 'MnM-TestRun-4-T10',
			# 'Plain': 'MnM-Plain', 
			# 'Plain-Aug': 'MnM-Plain-Aug', 
			# 'DE': 'MnM-DE-GoldStandard',  # This goes to final
			# 'LE': 'MnM-TestRun-1',  # resnet18, with Aug, 10 epochs
			# 'LE': 'MnM-TestRun-2',  # resnet18, no Aug, 200 epochs, STAPLE
			# 'LE-Aug': 'MnM-TestRun-3',  # resnet18, with Aug, 200 epochs, STAPLE
			# 'LE-All': 'MnM-TestRun-3-AllScanners',  # resnet18, with Aug, 200 epochs, STAPLE # This goes to final
			# 'LE-All-T5': 'MnM-TestRun-3-AllScanners-T5',
			# 'LE-CE': 'MnM-TestRun-4',  #  T7, resnet18, with Aug, 200 epochs, STAPLE

			# The following that have prediction depth info
			# '0.0': 'MnM-TestRun-4-T10-PD-0pCorr',
			# '0.3': 'MnM-TestRun-4-T10-PD-100pCorr',
			# '0.5': 'MnM-TestRun-4-T10-PD-100pCorr-0.5',
			# '0%': 'MnM-TestRun-4-T10-PD-0pCorr',
			# '50%': 'MnM-TestRun-4-T10-PD-50pCorr',
			# '100%': 'MnM-TestRun-4-T10-PD-100pCorr',
			# 'LE-0.0-Corrupted': 'MnM-TestRun-3-PredDepth',
			# 'LE-0.5-Corrupted': 'MnM-TestRun-3-PredDepth-50pCorrupted',
			# 'LE-1.0-Corrupted': 'MnM-TestRun-3-PredDepth-100pCorrupted',
		}
	else:
		raise ValueError(f"{dataset} not implemented")
	return experiments