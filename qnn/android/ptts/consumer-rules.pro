# JNI calls these classes, methods and constructors by name.
-keep class ai.gradium.phonon.QnnNative { *; }
-keep class ai.gradium.phonon.PhononTTS$Result { *; }
-keep interface ai.gradium.phonon.AudioCallback { *; }
-keepclassmembers class * implements ai.gradium.phonon.AudioCallback {
    public boolean onAudio(float[]);
}
