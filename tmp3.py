from moviepy.editor import VideoFileClip
import os
import sys

def convert_video_to_mp3(video_path, output_path=None):
    """
    Convert a video file to MP3 format
    :param video_path: Path to the input video file
    :param output_path: Path for the output MP3 file (optional)
    :return: Path of the created MP3 file
    """
    try:
        # If output path is not specified, create one based on input file
        if output_path is None:
            output_path = os.path.splitext(video_path)[0] + '.mp3'
        
        # Load the video file
        video = VideoFileClip(video_path)
        
        # Extract the audio
        audio = video.audio
        
        # Write the audio file
        audio.write_audiofile(output_path)
        
        # Close the video file to free up resources
        video.close()
        
        print(f"Successfully converted {video_path} to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python video_to_mp3.py <video_file_path> [output_path]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    convert_video_to_mp3(video_path, output_path) 