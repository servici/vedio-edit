from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip

def merge_audio_video(video_path, audio_path, output_path):
    # Load the video
    print("Loading video file...")
    video_clip = VideoFileClip(video_path)
    
    # Load the audio
    print("Loading audio file...")
    audio_clip = AudioFileClip(audio_path)
    
    # If audio duration is longer than video, trim it
    if audio_clip.duration > video_clip.duration:
        print("Trimming audio to match video duration...")
        audio_clip = audio_clip.subclipped(0, video_clip.duration)
    
    # Set the audio of the video clip
    print("Merging audio and video...")
    final_clip = video_clip.with_audio(audio_clip)
    
    # Write the result to a file
    print("Writing output file...")
    final_clip.write_videofile(output_path)
    
    # Close the clips to free up system resources
    print("Cleaning up...")
    video_clip.close()
    audio_clip.close()
    final_clip.close()
    
    print("Done! Output saved to:", output_path)

if __name__ == "__main__":
    video_file = "output_video.mp4"
    audio_file = "input_video.mp3"
    output_file = "final_output.mp4"
    
    print("Starting to merge audio and video...")
    merge_audio_video(video_file, audio_file, output_file) 