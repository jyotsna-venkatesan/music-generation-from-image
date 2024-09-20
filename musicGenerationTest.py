from midiutil import MIDIFile
import numpy as np
from PIL import Image
import colorsys
import random
import cv2
import sys
from scipy.stats import entropy
import scipy.ndimage as ndimage

print("Initiating cosmic soundscape generation with balanced volumes...")

def get_image_data(image_path):
    try:
        im = Image.open(image_path)
        rgb_array = np.array(im.convert('RGB').getdata(), dtype=np.float32) / 255.0
        hsv_array = np.array([colorsys.rgb_to_hsv(*pixel) for pixel in rgb_array])
        luminance = np.mean(rgb_array, axis=1)
        return rgb_array, hsv_array, luminance, im
    except Exception as e:
        print(f"Error in get_image_data: {e}")
        sys.exit(1)

def get_cosmic_tempo(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    entropy_value = entropy(gray.flatten())
    return int(100 + (entropy_value / 8) * 40)  # Map entropy to 100-140 bpm range

def get_stellar_time_signature(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    if lines is not None and len(lines) > 10:
        return 2  # More line-like shapes, 2/4
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    triangles = sum(1 for cnt in contours if len(cnt) == 3)
    return 3 if triangles > 5 else 4  # 3/4 if more triangles, else 4/4

def get_galactic_key_and_scale(hsv_array):
    hue_mean = np.mean(hsv_array[:, 0])
    value_mean = np.mean(hsv_array[:, 2])
    scales = ["C", "D", "E", "F", "G", "A", "B"]
    scale = scales[int(hue_mean * 7)]
    key = "major" if value_mean > 0.5 else "minor"
    return scale, key

def generate_cosmic_melody(edges, scale, key, length=64):
    base_note = {"C": 60, "D": 62, "E": 64, "F": 65, "G": 67, "A": 69, "B": 71}[scale]
    scale_intervals = [0, 2, 4, 5, 7, 9, 11] if key == "major" else [0, 2, 3, 5, 7, 8, 10]
    edge_density = np.mean(edges) / 255.0
    complexity = int(edge_density * 4) + 1  # 1 to 5
    durations = [0.25, 0.5, 1, 2][0:complexity]
    return [(base_note + random.choice(scale_intervals), random.choice(durations)) for _ in range(length)]

def create_nebula_chords(hsv_array, scale, key):
    hue_std = np.std(hsv_array[:, 0])
    if hue_std < 0.1:  # Analogous colors
        chords = [[0, 4, 7], [5, 9, 12], [7, 11, 14]] if key == "major" else [[0, 3, 7], [5, 8, 12], [7, 10, 14]]
    else:  # More complex for complementary colors
        chords = [[0, 4, 7], [5, 9, 12], [7, 11, 14], [3, 7, 10], [8, 12, 15]] if key == "major" else [[0, 3, 7], [5, 8, 12], [7, 10, 14], [3, 7, 10], [8, 11, 15]]
    base_note = {"C": 60, "D": 62, "E": 64, "F": 65, "G": 67, "A": 69, "B": 71}[scale]
    return [[note + base_note for note in chord] for chord in chords]

def select_cosmic_instrument(rgb_array):
    texture = np.std(rgb_array)
    return 4 if texture > 0.2 else 0  # Electric Piano if textured, else Acoustic Grand Piano

def get_cosmic_volume(luminance):
    return int(48 + luminance.mean() * 47)  # Map to MIDI volume range (48-95)

def generate_texture_based_piano(rgb_array, scale, key, length=64):
    base_note = {"C": 60, "D": 62, "E": 64, "F": 65, "G": 67, "A": 69, "B": 71}[scale]
    scale_intervals = [0, 2, 4, 5, 7, 9, 11] if key == "major" else [0, 2, 3, 5, 7, 8, 10]
    texture = np.std(rgb_array)
    complexity = int(texture * 5) + 1  # 1 to 6
    return [(base_note + random.choice(scale_intervals), random.uniform(0.5, 2.5), int(60 + random.uniform(-10, 10))) for _ in range(length)]

def generate_boutique_808(hsv_array, length=64):
    saturation = np.mean(hsv_array[:, 1])
    pattern_density = int(saturation * 8) + 1  # 1 to 9
    pattern = [36, 0, 0, 0, 38, 0, 0, 0][:pattern_density]
    return pattern * (length // len(pattern) + 1)

def generate_jitter_strings(rgb_array, scale, key, length=64):
    base_note = {"C": 60, "D": 62, "E": 64, "F": 65, "G": 67, "A": 69, "B": 71}[scale]
    scale_intervals = [0, 2, 4, 5, 7, 9, 11] if key == "major" else [0, 2, 3, 5, 7, 8, 10]
    jitter = np.std(rgb_array) * 5  # Reduced jitter
    return [(base_note + random.choice(scale_intervals), 0.25, int(48 + random.uniform(-jitter, jitter))) for _ in range(length)]

def generate_metro_bass(hsv_array, scale, length=64):
    base_note = {"C": 36, "D": 38, "E": 40, "F": 41, "G": 43, "A": 45, "B": 47}[scale]
    rhythm = [1, 0, 0.5, 0, 1, 0, 0.5, 0.5]
    intensity = np.mean(hsv_array[:, 2])  # Use value channel for intensity
    return [(base_note, duration, int(60 + intensity * 35)) if duration > 0 else (0, 0.25, 0) for duration in rhythm * (length // 8 + 1)]

def generate_dream_voice(rgb_array, scale, key, length=64):
    base_note = {"C": 72, "D": 74, "E": 76, "F": 77, "G": 79, "A": 81, "B": 83}[scale]
    scale_intervals = [0, 2, 4, 5, 7, 9, 11] if key == "major" else [0, 2, 3, 5, 7, 8, 10]
    dreaminess = 1 - np.std(rgb_array)  # More uniform colors = more dreamy
    return [(base_note + random.choice(scale_intervals), random.uniform(1, 4) * dreaminess, int(40 + random.uniform(0, 20))) for _ in range(length // 4)]

def generate_pulsating_waves(hsv_array, scale, length=64):
    base_note = {"C": 48, "D": 50, "E": 52, "F": 53, "G": 55, "A": 57, "B": 59}[scale]
    wave_speed = np.mean(hsv_array[:, 1])  # Use saturation for wave speed
    return [(base_note, 0.25, int(48 + 31 * np.sin(i * wave_speed))) for i in range(length)]


def create_cosmic_midi(image_path, output_file):
    try:
        rgb_array, hsv_array, luminance, im = get_image_data(image_path)
        tempo = get_cosmic_tempo(im)
        time_sig = get_stellar_time_signature(im)
        scale, key = get_galactic_key_and_scale(hsv_array)
        
        midi = MIDIFile(10)  # 10 tracks for all our cosmic elements
        midi.addTempo(0, 0, tempo)
        midi.addTimeSignature(0, 0, time_sig, 2, 24)
        
        base_volume = get_cosmic_volume(luminance)
        
        # Main Piano track
        piano_instrument = select_cosmic_instrument(rgb_array)
        midi.addProgramChange(0, 0, 0, piano_instrument)
        gray = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        melody = generate_cosmic_melody(edges, scale, key)
        for i, (note, duration) in enumerate(melody):
            midi.addNote(0, 0, note, i, duration, int(base_volume * 0.8))
        
        # Texture-based Piano track
        midi.addProgramChange(1, 0, 0, piano_instrument)
        texture_melody = generate_texture_based_piano(rgb_array, scale, key)
        time = 0
        for note, duration, velocity in texture_melody:
            midi.addNote(1, 0, note, time, duration, int(velocity * 0.7))
            time += duration
        
        # Boutique 808
        midi.addProgramChange(2, 9, 0, 0)  # Use channel 9 for percussion
        boutique_808 = generate_boutique_808(hsv_array)
        for i, note in enumerate(boutique_808):
            if note != 0:
                midi.addNote(2, 9, note, i * 0.25, 0.25, int(base_volume * 0.6))
        
        # Jitter Strings
        midi.addProgramChange(3, 0, 0, 49)  # String Ensemble 2
        jitter_strings = generate_jitter_strings(rgb_array, scale, key)
        for i, (note, duration, velocity) in enumerate(jitter_strings):
            midi.addNote(3, 0, note, i * 0.25, duration, int(velocity * 0.6))
        
        # Metro Bass
        midi.addProgramChange(4, 0, 0, 39)  # Synth Bass 2
        metro_bass = generate_metro_bass(hsv_array, scale)
        time = 0
        for note, duration, velocity in metro_bass:
            if note != 0:
                midi.addNote(4, 0, note, time, duration, int(velocity * 0.7))
            time += duration
        
        # Dream Voice
        midi.addProgramChange(5, 0, 0, 89)  # Pad 2 (warm)
        dream_voice = generate_dream_voice(rgb_array, scale, key)
        time = 0
        for note, duration, velocity in dream_voice:
            midi.addNote(5, 0, note, time, duration, int(velocity * 0.8))
            time += duration
        
        # Pulsating Waves
        midi.addProgramChange(6, 0, 0, 81)  # Lead 2 (sawtooth)
        pulsating_waves = generate_pulsating_waves(hsv_array, scale)
        for i, (note, duration, velocity) in enumerate(pulsating_waves):
            midi.addNote(6, 0, note, i * 0.25, duration, int(velocity * 0.5))
        
        # Nebula Chords
        midi.addProgramChange(7, 0, 0, 48)  # String Ensemble 1
        chords = create_nebula_chords(hsv_array, scale, key)
        for i, chord in enumerate(chords):
            for note in chord:
                midi.addNote(7, 0, note, i * 4, 4, int(base_volume * 0.5))
        
        # Bass track
        midi.addProgramChange(8, 0, 0, 33)  # Fingered Bass
        for i, chord in enumerate(chords):
            midi.addNote(8, 0, chord[0] - 12, i * 4, 4, int(base_volume * 0.7))
        
        # Additional Drums
        midi.addProgramChange(9, 9, 0, 0)  # Standard drum kit
        saturation = np.mean(hsv_array[:, 1])
        if saturation > 0.6:
            pattern = [36, 0, 38, 0, 36, 38, 36, 38]  # More complex
        elif saturation > 0.3:
            pattern = [36, 0, 38, 0, 36, 0, 38, 0]  # Medium
        else:
            pattern = [36, 0, 0, 0, 38, 0, 0, 0]  # Simple
        for i, drum in enumerate(pattern * 8):
            if drum != 0:
                midi.addNote(9, 9, drum, i * 0.5, 0.5, int(base_volume * 0.6))
        
        with open(output_file, "wb") as output_file:
            midi.writeFile(output_file)
        print(f"Balanced cosmic MIDI soundscape created: {output_file}")
        
    except Exception as e:
        print(f"Error in create_cosmic_midi: {e}")
        import traceback
        traceback.print_exc()

# Usage
print("Initiating cosmic soundscape generation with balanced volumes...")
try:
    create_cosmic_midi("nasaPicture2.jpg", "cosmic_output2_balanced.mid")
    print("Cosmic soundscape generation complete. Output saved to 'cosmic_output2_balanced.mid'.")
except Exception as e:
    print(f"An anomaly occurred during cosmic soundscape generation: {e}")