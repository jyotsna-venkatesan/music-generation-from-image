from midiutil import MIDIFile
import numpy as np
from PIL import Image
import os
import shutil
import traceback
import colorsys
import io
import random
import signal
import cv2
import sys
import logging

logging.basicConfig(level=logging.DEBUG)
print("Starting music generation...")

def get_image_data(image_path):
    """
    Load and process image data, converting to RGB and HSV arrays.
    
    Args:
        image_path (str): Path to the input image file.
    
    Returns:
        tuple: RGB array, HSV array, and warmth-coolness measure.
    """
    try:
        im = Image.open(image_path)
        rgb_array = np.array(im.convert('RGB').getdata(), dtype=np.float32) / 255.0
        hsv_array = np.array([colorsys.rgb_to_hsv(*pixel) for pixel in rgb_array])
        warmth_coolness = np.sum(rgb_array[:, :2], axis=1) - rgb_array[:, 2]
        return rgb_array, hsv_array, warmth_coolness
    except Exception as e:
        print(f"Error in get_image_data: {e}")
        sys.exit(1)

def get_tempo(rgb_array):
    """
    Calculate tempo based on image entropy.
    
    Args:
        rgb_array (np.array): RGB array of the image.
    
    Returns:
        int: Tempo value between 100 and 180 BPM.
    """
    entropy = -np.sum(rgb_array * np.log2(rgb_array + 1e-10))
    max_entropy = -np.log2(1/len(rgb_array)) * len(rgb_array)
    normalized_entropy = entropy / max_entropy
    return int(100 + normalized_entropy * 80)

def get_time_signature(image_path):
    """
    Determine time signature based on image complexity.
    
    Args:
        image_path (str): Path to the input image file.
    
    Returns:
        int: Time signature numerator (2 or 4).
    """
    img = cv2.imread(image_path, 0)
    edges = cv2.Canny(img, 100, 200)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
    return 2 if lines is not None and len(lines) > 10 else 4

def get_key_and_scale(hsv_array, warmth_coolness):
    """
    Determine musical key and scale based on image colors.
    
    Args:
        hsv_array (np.array): HSV array of the image.
        warmth_coolness (np.array): Warmth-coolness measure of the image.
    
    Returns:
        tuple: Musical scale and key (major/minor).
    """
    hue_bins = np.linspace(0, 1, 8)[:-1]
    most_common_hue = np.digitize(np.mean(hsv_array[:, 0]), hue_bins) - 1
    scale = ["C", "D", "E", "F", "G", "A", "B"][most_common_hue]
    key = "major" if np.mean(warmth_coolness) > 0 else "minor"
    return scale, key

def generate_melody(edges, scale, key):
    """
    Generate melody based on image edges and musical scale.
    
    Args:
        edges (np.array): Edge detection result of the image.
        scale (str): Musical scale.
        key (str): Musical key (major/minor).
    
    Returns:
        list: List of (note, duration) tuples representing the melody.
    """
    base_note = 60
    scale_intervals = [0, 2, 4, 5, 7, 9, 11] if key == "major" else [0, 2, 3, 5, 7, 8, 10]
    melody = []
    for edge in edges:
        note = base_note + random.choice(scale_intervals)
        duration = 0.25 if edge > 128 else 0.5
        melody.append((note, duration))
    return melody

def generate_chord_progression(hsv_array):
    """
    Generate chord progression based on color variance.
    
    Args:
        hsv_array (np.array): HSV array of the image.
    
    Returns:
        list: List of chord progressions.
    """
    color_variance = np.std(hsv_array[:, 0])
    return [0, 3, 4, 0] if color_variance < 0.1 else [0, 5, 3, 4]

def select_instrument(rgb_array):
    """
    Select MIDI instrument based on image texture.
    
    Args:
        rgb_array (np.array): RGB array of the image.
    
    Returns:
        int: MIDI instrument number.
    """
    texture = np.std(rgb_array)
    if texture < 0.1:
        return 1
    elif texture < 0.2:
        return 4
    else:
        return 26

def create_midi(image_path, output_file):
    """
    Create MIDI file from image.
    
    Args:
        image_path (str): Path to the input image file.
        output_file (str): Path to save the output MIDI file.
    """
    try:
        rgb_array, hsv_array, warmth_coolness = get_image_data(image_path)
        
        tempo = get_tempo(rgb_array)
        time_sig = get_time_signature(image_path)
        scale, key = get_key_and_scale(hsv_array, warmth_coolness)
        
        midi = MIDIFile(2)
        midi.addTempo(0, 0, tempo)
        
        instrument = select_instrument(rgb_array)
        midi.addProgramChange(0, 0, 0, instrument)
        
        img = cv2.imread(image_path, 0)
        edges = cv2.Canny(img, 100, 200)
        melody = generate_melody(edges.flatten(), scale, key)
        
        time = 0
        for note, duration in melody:
            if time + duration > 15:
                break
            midi.addNote(0, 0, note, time, duration, 100)
            time += duration
        
        midi.addProgramChange(1, 0, 0, 0)
        chord_prog = generate_chord_progression(hsv_array)
        
        time = 0
        for chord in chord_prog * 4:
            if time >= 15:
                break
            for note in [60 + chord, 64 + chord, 67 + chord]:
                midi.addNote(1, 0, note, time, 1, 80)
            time += 1

        midi_buffer = io.BytesIO()
        midi.writeFile(midi_buffer)

        def timeout_handler(signum, frame):
            raise TimeoutError("File writing operation timed out")

        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)

            directory = os.path.dirname(output_file) or '.'
            if not os.access(directory, os.W_OK):
                print(f"Error: No write permission in directory {directory}")
                return
            
            free_space = shutil.disk_usage(directory).free
            if free_space < 1024 * 1024:
                print(f"Warning: Low disk space. Only {free_space / (1024*1024):.2f} MB available.")

            signal.alarm(60)
            
            with open(output_file, "wb") as f:
                midi_buffer.seek(0)
                f.write(midi_buffer.getvalue())

            signal.alarm(0)

            if os.path.exists(output_file):
                print(f"MIDI file created successfully: {output_file}")
            else:
                print(f"Error: MIDI file was not created: {output_file}")

        except TimeoutError:
            print("Error: File writing operation timed out after 60 seconds")
        except PermissionError:
            print(f"Error: Permission denied when trying to write {output_file}")
        except IOError as e:
            print(f"IOError when writing file: {e}")
        except Exception as e:
            print(f"Unexpected error when writing file: {e}")
    
    except Exception as e:
        print(f"Error in create_midi: {e}")
        traceback.print_exc()

# Usage
print("Starting music generation process...")
try:
    create_midi("nasaPicture2.jpg", "output.mid")
    print("Music generation complete. Output saved to 'output.mid'.")
except Exception as e:
    print(f"An error occurred during music generation: {e}")