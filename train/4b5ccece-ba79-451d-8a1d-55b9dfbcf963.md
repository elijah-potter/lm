A tense hallway in an institutional building. Fluorescent lights buzz overhead, casting long shadows. **J** (mid-30s, sharp features, nervous energy) stands rigidly at the interrogation table, hands clenched, nails biting into palms. The chair opposite creaks as **R** (older, sharp eyes, a desk notebook open) slides toward the edge, pen poised.

**R:** *"You’re not here on a whim. Why now?"*

**J:** *(voice cracks slightly, voice modulator active)* "I was in a coma for six months. When I woke, I had no memory of my life before—no friends, no family. My old job, my apartment, my *self*—it’s all gone. I just… I don’t know how to explain it."

**R:** *"Explain what?"*

**J:** *(slams hands on the table, voice rising)* "I’m not a terrorist! I’m not a criminal! I’m not even *me*—I’m a ghost in my own life. I just… I don’t exist outside this room. I don’t *do* anything. I don’t *care* about anything. I’m just… here."

A beat. **R** scribbles something in the notebook. The door creaks open. **M** (young, wide-eyed, holding a file) steps in, camera flash cutting off the scene. **J**’s breath hits the table, a single, unsteady exhale. The moment hangs.

---
*(Output for downstream processing: JSON-like structure for parsing as a scene description)*
```json
{
  "scene": {
    "setting": {
      "location": "interrogation room, institutional building",
      "time": "night, institutional lighting",
      "atmosphere": "claustrophobic, high tension, fluorescent hum"
    },
    "characters": [
      {
        "role": "interrogator (R)",
        "actions": [
          "asks why they’re here",
          "opens notebook, takes notes",
          "slides chair toward table, pen poised"
        ],
        "dialogue": [
          { "text": "You’re not here on a whim. Why now?", "target": "J" }
        ]
      },
      {
        "role": "subject (J)",
        "actions": [
          "stands rigidly, hands clenched",
          "voice modulator active, cracked voice",
          "slams hands on table, rises in pitch",
          "slumps, voice drops to a whisper"
        ],
        "dialogue": [
          { "text": "I was in a coma for six months...", "target": "R" },
          { "text": "I’m not a terrorist! I’m not a criminal! I’m not even *me*—", "target": "R" },
          { "text": "I don’t exist outside this room...", "target": "R" }
        ]
      },
      {
        "role": "observer (M)",
        "actions": [
          "steps in with camera flash",
          "covers flash, door creaks"
        ]
      }
    ],
    "environment": {
      "objects": ["interrogation table", "fluorescent lights", "desk notebook", "camera flash"],
      "sound": ["buzzing lights", "creaking chair", "whispers, breath"]
    },
    "key_moments": [
      { "action": "J’s voice modulator activated", "effect": "disorienting, mechanical sound" },
      { "action": "R’s notebook open, pen poised", "effect": "preparation for recording" },
      { "action": "J’s breath hits table", "effect": "lasting visual: single exhalation" }
    ],
    "theme": ["identity loss", "institutional control", "psychological fragmentation"]
  },
  "notes": [
    "Visual: J’s posture shifts from rigid to slumped (subtle but telling)",
    "Audio: R’s pen scratches on notebook = deliberate tension",
    "Camera: Flash cuts off M’s face—intentional ambiguity"
  ]
}
```