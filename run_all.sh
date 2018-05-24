#!/bin/bash

rm ../Data/Videos/extracted_gros_plan/*
rm ../Data/Videos/extracted_plan_moyen/*
rm ../Data/Videos/extracted_plan_rapproche/*
rm ../Data/Videos/extracted_plan_large/*
rm ../Data/Videos/extracted_plan_americain/*

for video in ../Data/Videos/*.mp4; do
	[ -f "$video" ] || break
	./run_extract_frame.sh "${video%.*}" ../Data/Videos/extracted_gros_plan gros_plan
	./run_extract_frame.sh "${video%.*}" ../Data/Videos/extracted_plan_moyen plan_moyen
	./run_extract_frame.sh "${video%.*}" ../Data/Videos/extracted_plan_rapproche plan_rapproche
#	./run_extract_frame.sh "${video%.*}" ../Data/Videos/extracted_plan_americain plan_americain
#	./run_extract_frame.sh "${video%.*}" ../Data/Videos/extracted_plan_large plan_large
done
