import subprocess
import uuid
import os
import shutil

UPLOAD_DIR = "uploaded_audio"


class FFmpegError(Exception):
    """Custom exception for FFmpeg processing errors."""
    pass


def _run_ffmpeg(cmd):
    process = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    if process.returncode != 0:
        error_msg = process.stderr or "Unknown FFmpeg error"
        raise FFmpegError(f"FFmpeg command failed: {' '.join(cmd)}\nError: {error_msg}")
    
    return process


def apply_tempo(input_path, tempo_factor, output_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    try:
        tempo_factor = max(0.5, min(2.0, float(tempo_factor)))  # safe clamp
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-filter:a", f"atempo={tempo_factor}",
            output_path
        ]
        _run_ffmpeg(cmd)
        return output_path
    except FFmpegError:
        raise
    except Exception as e:
        raise FFmpegError(f"Tempo adjustment failed: {str(e)}")


def apply_eq(input_path, brightness_db, bass_db, output_path):

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    try:
        brightness_db = float(brightness_db)
        bass_db = float(bass_db)

        # Using firequalizer for clarity
        eq_filter = (
            f"firequalizer=gain_entry='entry(1000,{brightness_db})':"
            f"gain_entry='entry(100,{bass_db})'"
        )

        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-af", eq_filter,
            output_path
        ]
        _run_ffmpeg(cmd)
        return output_path
    except FFmpegError:
        raise
    except Exception as e:
        raise FFmpegError(f"EQ adjustment failed: {str(e)}")


def apply_reverb(input_path, reverb_amount, output_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Skip reverb if amount is too small
    if float(reverb_amount) < 0.01:
        # Just copy input to output
        shutil.copy2(input_path, output_path)
        return output_path
    
    try:
        reverb_amount = max(0.05, float(reverb_amount))
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-af", f"aecho=0.8:0.9:60:{reverb_amount}",
            output_path
        ]
        _run_ffmpeg(cmd)
        return output_path
    except FFmpegError:
        raise
    except Exception as e:
        raise FFmpegError(f"Reverb application failed: {str(e)}")


def apply_compression(input_path, amount, output_path):
    """
    Apply dynamic-range compression.
    
    Raises:
        FFmpegError: If FFmpeg processing fails
        FileNotFoundError: If input file doesn't exist
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Skip compression if amount is too small
    if float(amount) < 0.01:
        # Just copy input to output
        shutil.copy2(input_path, output_path)
        return output_path
    
    try:
        amount = float(amount)
        # Scale compression parameters based on amount (0.0 = no compression, 1.0 = heavy)
        ratio = 1.0 + (amount * 3.0)  # 1:1 to 4:1 ratio
        threshold = -20 - (amount * 10)  # -20dB to -30dB threshold
        
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-af", f"acompressor=threshold={threshold}dB:ratio={ratio}:attack=5:release=50",
            output_path
        ]
        _run_ffmpeg(cmd)
        return output_path
    except FFmpegError:
        raise
    except Exception as e:
        raise FFmpegError(f"Compression failed: {str(e)}")


def apply_variation_chain(input_path, params):
    """
    Apply all effects in sequence and return final output file path.
    
    Automatically cleans up intermediate files, even on errors.
    
    Args:
        input_path: Path to input WAV file
        params: Dictionary with transformation parameters:
            - tempo_factor: Tempo multiplier
            - brightness_db: Brightness EQ in dB
            - bass_db: Bass EQ in dB
            - reverb: Reverb amount 0-1
            - compression: Compression amount 0-1
    
    Returns:
        Path to final output WAV file
    
    Raises:
        FFmpegError: If any processing step fails
        FileNotFoundError: If input file doesn't exist
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    # Track intermediate files for cleanup
    intermediate_files = []
    final_path = None
    
    try:
        t1 = f"{UPLOAD_DIR}/{uuid.uuid4()}_tempo.wav"
        intermediate_files.append(t1)
        apply_tempo(input_path, params.get("tempo_factor", 1.0), t1)

        t2 = f"{UPLOAD_DIR}/{uuid.uuid4()}_eq.wav"
        intermediate_files.append(t2)
        apply_eq(t1, params.get("brightness_db", 0.0), params.get("bass_db", 0.0), t2)

        t3 = f"{UPLOAD_DIR}/{uuid.uuid4()}_reverb.wav"
        intermediate_files.append(t3)
        apply_reverb(t2, params.get("reverb", 0.0), t3)

        final_path = f"{UPLOAD_DIR}/{uuid.uuid4()}_final.wav"
        apply_compression(t3, params.get("compression", 0.0), final_path)

        return final_path
    
    except Exception:

        for temp_file in intermediate_files + ([final_path] if final_path else []):
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except OSError:
                    pass  # Ignore cleanup errors
        raise
    
    finally:
        for temp_file in intermediate_files:
            if os.path.exists(temp_file) and temp_file != final_path:
                try:
                    os.remove(temp_file)
                except OSError:
                    pass  # Ignore cleanup errors

