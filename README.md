# Cosmic Soundscape Generator

> Transform images into unique musical compositions using computer vision and algorithmic composition techniques.

## 🎵 Features

- **Image-to-Music Transformation**: Converts any image into a MIDI composition
- **Multi-track Compositions**: Generates 10 tracks of harmonically related musical elements
- **Visual-Musical Mapping**:
  - Image entropy → Tempo (100-140 BPM)
  - Shapes → Time signature (2/4, 3/4, 4/4)
  - Color temperature → Musical key (Major/Minor)
  - Color hue → Musical scale (C through B)
  - Edge detection → Melodic complexity
  - Color harmony → Chord progression complexity
  - Texture → Instrument selection
  - Brightness → Volume/intensity
  - Saturation → Rhythmic patterns

### Playing the Generated Music

The output is a standard MIDI file that can be played using:
- Virtual MIDI piano software (e.g., GarageBand, FL Studio)
- Online MIDI players
- Hardware MIDI devices

## 🔬 Technical Details

### Musical Element Generation

1. **Tempo Determination**
   - Image entropy calculates the randomness in the image
   - Higher entropy values produce faster tempos

2. **Time Signature Selection**
   - Uses computer vision to detect shapes
   - Line detection determines 2/4 time
   - Triangle detection determines 3/4 time
   - Default is 4/4 time

3. **Key and Scale Selection**
   - Hue values map to different musical scales (C through B)
   - Brightness determines major or minor key

4. **Melody Generation**
   - Edge detection creates the main melodic content
   - More edges result in more complex melodic patterns

5. **Chord Progression Creation**
   - Analyzes color relationships
   - Analogous colors create simpler progressions
   - Complementary colors create more complex progressions

6. **Instrument Selection**
   - Based on texture analysis
   - Grainier textures use Electric Piano
   - Smoother textures use Acoustic Grand Piano

### Track Overview

The generated MIDI file contains 10 tracks:
1. Main Piano - Carries the primary melody
2. Texture Piano - Adds complementary melodic elements
3. Boutique 808 - Percussion elements based on image saturation
4. Jitter Strings - String elements with subtle variations
5. Metro Bass - Bass patterns following the harmonic structure
6. Dream Voice - Pad sounds that match the image "dreaminess"
7. Pulsating Waves - Lead sounds with dynamic volume changes
8. Nebula Chords - Chord progressions matching color harmony
9. Bass - Fundamental bass notes supporting the harmony
10. Additional Drums - Rhythm patterns matching image saturation


## 📄 License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute it, including for commercial purposes, with proper attribution.

