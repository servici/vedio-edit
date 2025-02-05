import torch
import torchaudio
import numpy as np
import cv2
from pathlib import Path

def apply_audio_effects(audio_waveform):
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

def process_video_audio(video_path, sample_rate=44100):
    """Extract and process audio from video"""
    print("Extracting audio from video...")
    # Load video and extract audio using torchaudio
    try:
        audio, sr = torchaudio.load(video_path)
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            audio = resampler(audio)
        
        # Convert to stereo if mono
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        
        # Apply audio effects
        print("Applying audio effects...")
        processed_audio = apply_audio_effects(audio)
        
        # Save processed audio
        torchaudio.save("processed_video_audio.wav", processed_audio, sample_rate)
        print("Processed video audio saved as processed_video_audio.wav")
        return processed_audio
    except Exception as e:
        print(f"Warning: Could not extract audio from video: {str(e)}")
        return None

def generate_ai_music(duration=10, sample_rate=44100):
    print("Generating AI music...")
    # Generate a simple melody using sine waves
    t = torch.linspace(0, duration, int(duration * sample_rate))
    
    # Create harmonies
    frequencies = [
        [440, 550, 660, 880],  # A4, C#5, E5, A5 (main chord)
        [523, 659, 784],       # C5, E5, G5 (second chord)
    ]
    
    waveform = torch.zeros_like(t)
    
    # Add multiple layers of sounds
    for chord in frequencies:
        for freq in chord:
            # Basic wave
            wave = torch.sin(2 * np.pi * freq * t)
            # Add harmonics
            wave += 0.5 * torch.sin(4 * np.pi * freq * t)
            wave += 0.25 * torch.sin(6 * np.pi * freq * t)
            
            # Dynamic envelope
            envelope = 0.5 * (1 + torch.sin(2 * np.pi * 0.2 * t))
            waveform += wave * envelope
    
    # Normalize
    waveform = waveform / torch.max(torch.abs(waveform))
    # Convert to stereo
    waveform = waveform.unsqueeze(0).repeat(2, 1)
    
    # Apply effects
    waveform = apply_audio_effects(waveform)
    
    print("Saving AI music...")
    torchaudio.save("ai_background_music.wav", waveform, sample_rate)
    print("AI music saved as ai_background_music.wav")
    return waveform

def mix_audio(audio1, audio2, mix_ratio=0.7):
    """Mix two audio tracks"""
    if audio1 is None or audio2 is None:
        return audio1 if audio1 is not None else audio2
        
    # Ensure both have the same length
    min_length = min(audio1.shape[1], audio2.shape[1])
    audio1 = audio1[:, :min_length]
    audio2 = audio2[:, :min_length]
    
    # Mix the audio
    mixed = mix_ratio * audio1 + (1 - mix_ratio) * audio2
    
    # Normalize
    mixed = mixed / torch.max(torch.abs(mixed))
    return mixed

def apply_ai_color_grading(frame):
    """Apply AI-enhanced color grading"""
    # Convert to float32 for better precision
    frame = frame.astype(np.float32) / 255.0
    
    # Enhance colors
    # Increase vibrancy
    saturation = 1.3
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame[:, :, 1] = frame[:, :, 1] * saturation
    frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
    
    # Apply cinematic color grading
    # Enhance shadows and highlights
    shadows = frame * 0.1
    highlights = np.clip(frame * 1.5, 0, 1)
    frame = np.clip(frame * 1.2 + shadows * 0.6 + highlights * 0.4, 0, 1)
    
    # Add subtle vignette effect
    rows, cols = frame.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols/4)
    kernel_y = cv2.getGaussianKernel(rows, rows/4)
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    frame = frame * mask[:, :, np.newaxis]
    
    # Convert back to uint8
    return (frame * 255).astype(np.uint8)

def edit_video(input_video_path, output_video_path="output_video.mp4"):
    print("Loading video...")
    cap = cv2.VideoCapture(input_video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    print("Processing video...")
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply AI color grading
        frame = apply_ai_color_grading(frame)
        
        # Write the frame
        out.write(frame)
        frame_count += 1
        
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")
    
    # Release everything
    cap.release()
    out.release()
    print(f"Video saved as {output_video_path}")

if __name__ == "__main__":
    try:
        input_video = "input_video.mp4"
        
        # Process original video audio with effects
        processed_audio = process_video_audio(input_video)
        
        # Generate AI music with effects
        ai_music = generate_ai_music()
        
        # Mix processed audio with AI music
        if processed_audio is not None:
            print("Mixing processed audio with AI music...")
            final_audio = mix_audio(processed_audio, ai_music, mix_ratio=0.7)
            torchaudio.save("final_mixed_audio.wav", final_audio, 44100)
            print("Final mixed audio saved as final_mixed_audio.wav")
        
        # Edit video with AI color grading
        edit_video(input_video)
        
        print("\nProcessing completed successfully!")
        print("Generated files:")
        print("1. processed_video_audio.wav - Original video audio with effects")
        print("2. ai_background_music.wav - AI generated music with effects")
        print("3. final_mixed_audio.wav - Mixed original and AI audio")
        print("4. output_video.mp4 - Color graded video")
        print("\nPlease use a video editor to combine output_video.mp4 with final_mixed_audio.wav")
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please make sure you have a video file named 'input_video.mp4' in the same directory.")