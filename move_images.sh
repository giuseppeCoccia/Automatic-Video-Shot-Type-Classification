#!/bin/sh

rm ../Data/Training/Gros\ plan/*
rm ../Data/Training/Plan\ moyen/*
rm ../Data/Training/Plan\ rapproche/*
rm ../Data/Test/Gros\ plan/*
rm ../Data/Test/Plan\ moyen/*
rm ../Data/Test/Plan\ rapproche/*

find ../Data/Videos/extracted_gros_plan/ -maxdepth 1 -type f | head -$1 | xargs cp -t ../Data/Training/Gros\ plan/
find ../Data/Videos/extracted_gros_plan/ -maxdepth 1 -type f | tail -$2 | xargs cp -t ../Data/Test/Gros\ plan/

find ../Data/Videos/extracted_plan_moyen/ -maxdepth 1 -type f | head -$1 | xargs cp -t ../Data/Training/Plan\ moyen/
find ../Data/Videos/extracted_plan_moyen/ -maxdepth 1 -type f | tail -$2 | xargs cp -t ../Data/Test/Plan\ moyen/

find ../Data/Videos/extracted_plan_rapproche/ -maxdepth 1 -type f | head -$1 | xargs cp -t ../Data/Training/Plan\ rapproche/
find ../Data/Videos/extracted_plan_rapproche/ -maxdepth 1 -type f | tail -$2 | xargs cp -t ../Data/Test/Plan\ rapproche/
