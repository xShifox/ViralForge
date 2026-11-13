﻿Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.SelectVoice('Microsoft David Desktop')
$s.Rate = 1
$s.Volume = 100
$path = '<RELEASE_ROOT>\data\logs\e2e-speech.wav'
$s.SetOutputToWaveFile($path)
$script = "Did you know that the average human brain generates more electrical impulses in a single second than all the telephones in the world generate in an entire minute? Nobody talks about this, but it is the single most misunderstood fact in modern neuroscience. Here is what actually happens inside your head when you think a single thought. First, the electrical signal travels along a neuron at over two hundred miles per hour. Second, your brain burns twenty percent of your body's entire energy budget just keeping the lights on. Third, and this is the part that shocks most people, your brain consumes almost no extra energy when you actually use it. The reason is simple and beautiful. Your brain is quietly training a prediction machine every waking moment of your life, and it updates that machine billions of times per day without you ever noticing. So the next time someone says they have a brain like everyone else, remember this one fact. Your brain is the most complex object ever discovered in the observable universe, and it weighs less than three pounds. That is the miracle we all carry around every single second of every single day."
$s.Speak($script)
$s.SetOutputToDefaultAudioDevice()
$s.Dispose()
