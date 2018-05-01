#!/bin/bash

rm ../Data/Videos/extracted_frames_gros_plan/*
rm ../Data/Videos/extracted_frames_plan_moyen/*
rm ../Data/Videos/extracted_frames_plan_rapproche/*

for video in ../Data/Videos/*.mp4; do
	[ -f "$video" ] || break
	./run_extract_frame.sh "${video%.*}" ../Data/Videos/extracted_frames_gros_plan gros_plan
	./run_extract_frame.sh "${video%.*}" ../Data/Videos/extracted_frames_plan_moyen plan_moyen
	./run_extract_frame.sh "${video%.*}" ../Data/Videos/extracted_frames_plan_rapproche plan_rapproche
done
