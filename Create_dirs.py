#!/usr/bin/env python
import os
current_directory = os.getcwd()
for dir in ['step1','step2','step3','step4','step5']:
	final_directory = os.path.join(current_directory, dir)
	if not os.path.exists(final_directory):
		os.makedirs(final_directory)
