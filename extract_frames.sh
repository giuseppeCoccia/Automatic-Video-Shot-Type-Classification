# path to video
input_=$1
# path to output dir
output_=$2

read frames_in
echo $frames_in
# if input is empty exit
if [ -z $frames_in ]; then
	echo "No entry"
	exit 1
fi

basename_=$(basename $input_)
#selected_string format: 'eq(n\,100)+eq(n\,184)+...'
for i in $frames_in
do
	frame="eq(n\,$i)"
	ffmpeg -i "$input_" -vf select=$frame -vsync 0 "$output_/${basename_%.*}_$i.jpg"
done
