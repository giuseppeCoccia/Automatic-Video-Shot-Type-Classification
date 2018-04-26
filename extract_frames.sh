# path to video
input_=$1
# path to output dir
output_=$2

read frames_in

# if input is empty exit
if [ -z $frames_in ]; then
	echo "No entry"
	exit 1
fi

frames=''
#selected_string format: 'eq(n\,100)+eq(n\,184)+...'
#for i in "${@:2}"
for i in $frames_in
do
	frames+="eq(n\,$i)+"
done

#delete last char
frames=${frames::-1}

basename_=$(basename $input_)
ffmpeg -i "$input_" -vf select=$frames -vsync 0 "$output_/${basename_%.*}_%d.jpg"
