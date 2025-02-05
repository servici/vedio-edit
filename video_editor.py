import torch
import torchaudio
import numpy as np
import cv2
from pathlib import Path
import os
import sys
from moviepy.editor import VideoFileClip, AudioFileClip

class VideoEditor:
    def __init__(self):
        self.sample_rate = 44100
        # Create temp directory if it doesn't exist
        self.temp_dir = "temp"
        os.makedirs(self.temp_dir, exist_ok=True)

    def get_temp_path(self, filename):
        """Get path for temporary files"""
        return os.path.join(self.temp_dir, filename)

    def apply_audio_effects(self, audio_waveform):
        """Apply simple audio effects"""
        # Add echo effect
        echo_delay = 22050  # 0.5 second delay at 44100 Hz
        echo = torch.zeros_like(audio_waveform)
        echo[:, echo_delay:] = audio_waveform[:, :-echo_delay] * 0.6
        
        # Mix original with echo
        audio_waveform = audio_waveform + echo
        
        # Add tremolo effect (amplitude modulation)
        t = torch.linspace(0, audio_waveform.shape[1]/44100, audio_waveform.shape[1])
        tremolo = 0.7 + 0.3 * torch.sin(2 * np.pi * 5 * t)  # 5 Hz tremolo
        audio_waveform = audio_waveform * tremolo
        
        # Process each channel separately for the low-pass filter
        filtered_audio = torch.zeros_like(audio_waveform)
        kernel_size = 5
        kernel = torch.ones(1, 1, kernel_size) / kernel_size
        
        for i in range(audio_waveform.shape[0]):
            filtered_audio[i:i+1] = torch.nn.functional.conv1d(
                audio_waveform[i:i+1].unsqueeze(0),
                kernel,
                padding=kernel_size//2
            ).squeeze(0)
        
        # Normalize
        filtered_audio = filtered_audio / torch.max(torch.abs(filtered_audio))
        return filtered_audio

    def convert_video_to_mp3(self, video_path, output_path=None):
        """Convert a video file to MP3 format"""
        try:
            if output_path is None:
                output_path = self.get_temp_path(os.path.splitext(os.path.basename(video_path))[0] + '.mp3')
            
            video = VideoFileClip(video_path)
            audio = video.audio
            audio.write_audiofile(output_path)
            video.close()
            
            print(f"Successfully converted {video_path} to {output_path}")
            return output_path
            
        except Exception as e:
            print(f"An error occurred during conversion: {str(e)}")
            return None

    def merge_audio_video(self, video_path, audio_path, output_path):
        """Merge audio and video files"""
        print("Loading video file...")
        video_clip = VideoFileClip(video_path)
        
        print("Loading audio file...")
        audio_clip = AudioFileClip(audio_path)
        
        if audio_clip.duration > video_clip.duration:
            print("Trimming audio to match video duration...")
            audio_clip = audio_clip.subclipped(0, video_clip.duration)
        
        print("Merging audio and video...")
        final_clip = video_clip.with_audio(audio_clip)
        
        print("Writing output file...")
        final_clip.write_videofile(output_path)
        
        print("Cleaning up...")
        video_clip.close()
        audio_clip.close()
        final_clip.close()
        
        print("Done! Output saved to:", output_path)

    def process_video_audio(self, video_path):
        """Extract and process audio from video"""
        print("Extracting audio from video...")
        try:
            audio, sr = torchaudio.load(video_path)
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                audio = resampler(audio)
            
            if audio.shape[0] == 1:
                audio = audio.repeat(2, 1)
            
            print("Applying audio effects...")
            processed_audio = self.apply_audio_effects(audio)
            
            output_path = self.get_temp_path("processed_video_audio.wav")
            torchaudio.save(output_path, processed_audio, self.sample_rate)
            print("Processed video audio saved as", output_path)
            return output_path
        except Exception as e:
            print(f"Warning: Could not extract audio from video: {str(e)}")
            return None

    def apply_ai_color_grading(self, frame, effects):
        """Apply AI-enhanced color grading with customizable effects"""
        frame = frame.astype(np.float32) / 255.0
        intensity = float(effects['intensity']) / 100.0  # Convert to 0-1 range
        
        if effects['saturation']:
            # Enhance colors with customizable intensity
            saturation = 1 + (0.6 * intensity)  # Range: 1.0 to 1.6
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            frame[:, :, 1] = frame[:, :, 1] * saturation
            frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
        
        if effects['contrast']:
            # Apply cinematic color grading with customizable intensity
            shadows = frame * (0.1 * intensity)
            highlights = np.clip(frame * (1 + intensity), 0, 1)
            frame = np.clip(frame * (1 + (0.4 * intensity)) + shadows * (0.6 * intensity) + highlights * (0.4 * intensity), 0, 1)
        
        if effects['vignette']:
            # Add subtle vignette effect with customizable intensity
            rows, cols = frame.shape[:2]
            kernel_x = cv2.getGaussianKernel(cols, cols/4)
            kernel_y = cv2.getGaussianKernel(rows, rows/4)
            kernel = kernel_y * kernel_x.T
            mask = kernel / kernel.max()
            
            # Adjust vignette intensity
            mask = 1 - ((1 - mask) * intensity)
            frame = frame * mask[:, :, np.newaxis]
        
        return (frame * 255).astype(np.uint8)

    def process_video(self, input_video_path, output_path, effects=None):
        """Process video with color grading while keeping original audio"""
        if effects is None:
            effects = {
                'saturation': True,
                'contrast': True,
                'vignette': True,
                'intensity': 70
            }

        print("Loading video...")
        # First create the color graded video without audio
        cap = cv2.VideoCapture(input_video_path)
        temp_video_path = self.get_temp_path("temp_video.mp4")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
        
        print("Applying color grading...")
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = self.apply_ai_color_grading(frame, effects)
            out.write(frame)
            frame_count += 1
            
            if frame_count % 30 == 0:  # Update more frequently
                progress = int((frame_count / total_frames) * 100)
                print(f"Processed {frame_count} frames ({progress}%)...")
        
        cap.release()
        out.release()
        
        # Now combine the color graded video with the original audio
        print("Combining with original audio...")
        original_video = VideoFileClip(input_video_path)
        graded_video = VideoFileClip(temp_video_path)
        
        # Set the audio from the original video
        final_video = graded_video.set_audio(original_video.audio)
        
        # Write the final video
        print("Saving final video...")
        final_video.write_videofile(output_path)
        
        # Clean up
        original_video.close()
        graded_video.close()
        final_video.close()
        
        # Remove temporary file
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        
        print(f"Video processing completed. Saved to: {output_path}")
        return output_path

def main():
    editor = VideoEditor()
    
    try:
        # Example usage
        input_video = "input_video.mp4"
        
        # 1. Convert video to MP3
        print("\n1. Converting video to MP3...")
        editor.convert_video_to_mp3(input_video)
        
        # 2. Process video audio with effects
        print("\n2. Processing video audio...")
        processed_audio = editor.process_video_audio(input_video)
        
        # 3. Edit video with AI color grading
        print("\n3. Editing video...")
        editor.process_video(input_video, "output_video.mp4")
        
        # 4. Merge processed video with original audio
        print("\n4. Merging final video and audio...")
        editor.merge_audio_video("output_video.mp4", "processed_video_audio.wav", "final_output.mp4")
        
        print("\nProcessing completed successfully!")
        print("Generated files:")
        print("1. input_video.mp3 - Extracted audio")
        print("2. processed_video_audio.wav - Original video audio with effects")
        print("3. output_video.mp4 - Color graded video")
        print("4. final_output.mp4 - Final video with original audio")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please make sure you have a video file named 'input_video.mp4' in the same directory.")

if __name__ == "__main__":
    main() 