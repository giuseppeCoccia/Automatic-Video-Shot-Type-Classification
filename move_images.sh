#!/bin/sh

path_to_dir=$4

rm $path_to_dir/Training/Gros\ plan/*
rm $path_to_dir/Training/Plan\ moyen/*
rm $path_to_dir/Training/Plan\ rapproche/*

rm $path_to_dir/Validation/Gros\ plan/*
rm $path_to_dir/Validation/Plan\ moyen/*
rm $path_to_dir/Validation/Plan\ rapproche/*

rm $path_to_dir/Test/Gros\ plan/*
rm $path_to_dir/Test/Plan\ moyen/*
rm $path_to_dir/Test/Plan\ rapproche/*

	
find ../Data/Videos/extracted_gros_plan/ -maxdepth 1 -type f | awk -v t=$1 'NR <= t' | xargs cp -t $path_to_dir/Training/Gros\ plan/
find ../Data/Videos/extracted_plan_moyen/ -maxdepth 1 -type f | awk -v t=$1 'NR <= t' | xargs cp -t $path_to_dir/Training/Plan\ moyen/
find ../Data/Videos/extracted_plan_rapproche/ -maxdepth 1 -type f | awk -v t=$1 'NR <= t' | xargs cp -t $path_to_dir/Training/Plan\ rapproche/

find ../Data/Videos/extracted_gros_plan/ -maxdepth 1 -type f | awk -v t=$1 -v v=$2 'NR > t && NR <= v' | xargs cp -t $path_to_dir/Validation/Gros\ plan/
find ../Data/Videos/extracted_plan_moyen/ -maxdepth 1 -type f | awk -v t=$1 -v v=$2 'NR > t && NR <= v' | xargs cp -t $path_to_dir/Validation/Plan\ moyen/
find ../Data/Videos/extracted_plan_rapproche/ -maxdepth 1 -type f | awk -v t=$1 -v v=$2 'NR > t && NR <= v'  | xargs cp -t $path_to_dir/Validation/Plan\ rapproche/

find ../Data/Videos/extracted_gros_plan/ -maxdepth 1 -type f | awk -v v=$2 -v t=$3 'NR > v && NR <= t' | xargs cp -t $path_to_dir/Test/Gros\ plan/
find ../Data/Videos/extracted_plan_moyen/ -maxdepth 1 -type f | awk -v v=$2 -v t=$3 'NR > v && NR <= t' | xargs cp -t $path_to_dir/Test/Plan\ moyen/
find ../Data/Videos/extracted_plan_rapproche/ -maxdepth 1 -type f | awk -v v=$2 -v t=$3 'NR > v && NR <= t' | xargs cp -t $path_to_dir/Test/Plan\ rapproche/
