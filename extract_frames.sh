input_=$1
read frames_in
echo $frames_in
frames=''
#selected_string format: 'eq(n\,100)+eq(n\,184)+...'
#for i in "${@:2}"
for i in $frames_in
do
	frames+="eq(n\,$i)+"
done

#delete last char
frames=${frames::-1}

ffmpeg -i "/datas/teaching/projects/spring2018/ps34/Data/Videos/${input_}.mp4" -vf select=$frames -vsync 0 "/datas/teaching/projects/spring2018/ps34/Data/Videos/extracted_frames/${input_%.*}_%d.jpg"
