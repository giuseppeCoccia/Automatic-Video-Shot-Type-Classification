#!/bin/sh

rm ../Data/Training/Gros\ plan/*
rm ../Data/Training/Plan\ moyen/*
rm ../Data/Training/Plan\ rapproche/*

rm ../Data/Validation/Gros\ plan/*
rm ../Data/Validation/Plan\ moyen/*
rm ../Data/Validation/Plan\ rapproche/*

rm ../Data/Test/Gros\ plan/*
rm ../Data/Test/Plan\ moyen/*
rm ../Data/Test/Plan\ rapproche/*

	
find ../Data/Videos/extracted_gros_plan/ -maxdepth 1 -type f | awk -v t=$1 'NR <= t' | xargs cp -t ../Data/Training/Gros\ plan/
find ../Data/Videos/extracted_plan_moyen/ -maxdepth 1 -type f | awk -v t=$1 'NR <= t' | xargs cp -t ../Data/Training/Plan\ moyen/
find ../Data/Videos/extracted_plan_rapproche/ -maxdepth 1 -type f | awk -v t=$1 'NR <= t' | xargs cp -t ../Data/Training/Plan\ rapproche/

find ../Data/Videos/extracted_gros_plan/ -maxdepth 1 -type f | awk -v t=$1 -v v=$2 'NR > t && NR <= v' | xargs cp -t ../Data/Validation/Gros\ plan/
find ../Data/Videos/extracted_plan_moyen/ -maxdepth 1 -type f | awk -v t=$1 -v v=$2 'NR > t && NR <= v' | xargs cp -t ../Data/Validation/Plan\ moyen/
find ../Data/Videos/extracted_plan_rapproche/ -maxdepth 1 -type f | awk -v t=$1 -v v=$2 'NR > t && NR <= v'  | xargs cp -t ../Data/Validation/Plan\ rapproche/

find ../Data/Videos/extracted_gros_plan/ -maxdepth 1 -type f | awk -v v=$2 -v t=$3 'NR > v && NR <= t' | xargs cp -t ../Data/Test/Gros\ plan/
find ../Data/Videos/extracted_plan_moyen/ -maxdepth 1 -type f | awk -v v=$2 -v t=$3 'NR > v && NR <= t' | xargs cp -t ../Data/Test/Plan\ moyen/
find ../Data/Videos/extracted_plan_rapproche/ -maxdepth 1 -type f | awk -v v=$2 -v t=$3 'NR > v && NR <= t' | xargs cp -t ../Data/Test/Plan\ rapproche/
