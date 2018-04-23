input_=$1

frames=''
#selected_string format: 'eq(n\,100)+eq(n\,184)+...'
for i in "${@:2}"
do
	frames+="eq(n\,$i)+"
done

#delete last char
frames=${frames::-1}

ffmpeg -i $input_ -vf select=$frames -vsync 0 "${input_%.*}_%d.jpg"
