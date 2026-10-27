#include "phonon.hpp"
#include <atomic>
#include <cstdlib>
#include <jni.h>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

namespace {
struct Session {
  std::unique_ptr<phonon::Phonon> model;
  std::atomic<bool> stopped{false};
};
Session &session(jlong handle) {
  if (!handle)
    throw std::runtime_error("PhononTTS is closed");
  return *reinterpret_cast<Session *>(handle);
}
void error(JNIEnv *env, const std::exception &e) {
  if (!env->ExceptionCheck()) {
    auto cls = env->FindClass("java/lang/IllegalStateException");
    env->ThrowNew(cls, e.what());
    env->DeleteLocalRef(cls);
  }
}
// JNI's modified UTF-8 would change supplementary characters before
// tokenization.
std::string utf8(JNIEnv *env, jstring value) {
  if (!value)
    throw std::runtime_error("null string");
  auto cls = env->FindClass("java/lang/String");
  auto method = env->GetMethodID(cls, "getBytes", "(Ljava/lang/String;)[B");
  auto charset = env->NewStringUTF("UTF-8");
  auto bytes =
      static_cast<jbyteArray>(env->CallObjectMethod(value, method, charset));
  env->DeleteLocalRef(charset);
  env->DeleteLocalRef(cls);
  if (!bytes || env->ExceptionCheck())
    throw std::runtime_error("cannot encode string");
  std::string out(env->GetArrayLength(bytes), '\0');
  env->GetByteArrayRegion(bytes, 0, out.size(),
                          reinterpret_cast<jbyte *>(out.data()));
  env->DeleteLocalRef(bytes);
  if (env->ExceptionCheck())
    throw std::runtime_error("cannot read string");
  if (out.find('\0') != std::string::npos)
    throw std::runtime_error("strings cannot contain NUL");
  return out;
}
jstring java_string(JNIEnv *env, const std::string &text) {
  auto cls = env->FindClass("java/lang/String");
  auto ctor = env->GetMethodID(cls, "<init>", "([BLjava/lang/String;)V");
  auto bytes = env->NewByteArray(text.size());
  if (!bytes)
    throw std::bad_alloc();
  env->SetByteArrayRegion(bytes, 0, text.size(),
                          reinterpret_cast<const jbyte *>(text.data()));
  auto charset = env->NewStringUTF("UTF-8");
  auto out = static_cast<jstring>(env->NewObject(cls, ctor, bytes, charset));
  env->DeleteLocalRef(charset);
  env->DeleteLocalRef(bytes);
  env->DeleteLocalRef(cls);
  return out;
}
} // namespace

extern "C" JNIEXPORT jlong JNICALL Java_ai_gradium_phonon_QnnNative_create(
    JNIEnv *env, jobject, jstring bundle, jstring lang, jstring soc,
    jstring libraries) {
  try {
    phonon::Options options;
    options.bundle_dir = utf8(env, bundle);
    options.lang = utf8(env, lang);
    options.soc_model = utf8(env, soc);
    options.lib_dir = utf8(env, libraries);
    options.text_library = options.lib_dir + "/libptts_text.so";
    // The extracted APK libraries contain the matching DSP skels too. Configure
    // before QNN loads.
    static std::mutex init;
    std::lock_guard<std::mutex> lock(init);
    const char *existing = std::getenv("ADSP_LIBRARY_PATH");
    std::string dsp = existing ? existing : "";
    if (dsp.empty())
      dsp = "/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/system/lib/rfsa/adsp;/dsp";
    const std::string prefix = options.lib_dir + ";";
    if (dsp != options.lib_dir && dsp.rfind(prefix, 0) != 0)
      dsp = prefix + dsp;
    if (setenv("ADSP_LIBRARY_PATH", dsp.c_str(), 1) != 0)
      throw std::runtime_error("cannot set ADSP_LIBRARY_PATH");
    auto state = std::make_unique<Session>();
    state->model = std::make_unique<phonon::Phonon>(options);
    return reinterpret_cast<jlong>(state.release());
  } catch (const std::exception &e) {
    error(env, e);
    return 0;
  }
}
extern "C" JNIEXPORT jint JNICALL Java_ai_gradium_phonon_QnnNative_sampleRate(
    JNIEnv *env, jobject, jlong handle) {
  try {
    return session(handle).model->sample_rate();
  } catch (const std::exception &e) {
    error(env, e);
    return 0;
  }
}
extern "C" JNIEXPORT jobjectArray JNICALL
Java_ai_gradium_phonon_QnnNative_voices(JNIEnv *env, jobject, jlong handle) {
  try {
    auto names = session(handle).model->voices();
    auto cls = env->FindClass("java/lang/String");
    auto out = env->NewObjectArray(names.size(), cls, nullptr);
    env->DeleteLocalRef(cls);
    if (!out)
      throw std::bad_alloc();
    for (size_t i = 0; i < names.size(); ++i) {
      auto name = java_string(env, names[i]);
      env->SetObjectArrayElement(out, i, name);
      env->DeleteLocalRef(name);
      if (env->ExceptionCheck())
        return nullptr;
    }
    return out;
  } catch (const std::exception &e) {
    error(env, e);
    return nullptr;
  }
}
extern "C" JNIEXPORT void JNICALL
Java_ai_gradium_phonon_QnnNative_resetStop(JNIEnv *env, jobject, jlong handle) {
  try {
    session(handle).stopped.store(false);
  } catch (const std::exception &e) {
    error(env, e);
  }
}
extern "C" JNIEXPORT void JNICALL
Java_ai_gradium_phonon_QnnNative_stop(JNIEnv *env, jobject, jlong handle) {
  try {
    session(handle).stopped.store(true);
  } catch (const std::exception &e) {
    error(env, e);
  }
}
extern "C" JNIEXPORT jobject JNICALL Java_ai_gradium_phonon_QnnNative_speak(
    JNIEnv *env, jobject, jlong handle, jstring text, jstring voice,
    jobject callback) {
  try {
    auto &state = session(handle);
    auto prompt = utf8(env, text);
    auto selected = voice ? utf8(env, voice) : state.model->default_voice();
    auto cls = env->GetObjectClass(callback);
    auto on_audio = env->GetMethodID(cls, "onAudio", "([F)Z");
    env->DeleteLocalRef(cls);
    if (!on_audio)
      return nullptr;
    phonon::Stats stats;
    state.model->synthesize(
        prompt, selected,
        [&](const float *pcm, size_t count) {
          if (state.stopped.load())
            return false;
          auto array = env->NewFloatArray(count);
          if (!array)
            return false;
          env->SetFloatArrayRegion(array, 0, count, pcm);
          bool keep = false;
          if (!env->ExceptionCheck())
            keep = env->CallBooleanMethod(callback, on_audio, array);
          env->DeleteLocalRef(array);
          return keep && !env->ExceptionCheck();
        },
        &stats, [&] { return state.stopped.load(); }, false);
    if (env->ExceptionCheck())
      return nullptr;
    auto result_class = env->FindClass("ai/gradium/phonon/PhononTTS$Result");
    auto ctor = env->GetMethodID(result_class, "<init>", "(IJDDZ)V");
    auto result =
        env->NewObject(result_class, ctor, stats.frames,
                       static_cast<jlong>(stats.samples), stats.first_audio_ms,
                       stats.total_ms, static_cast<jboolean>(stats.cancelled));
    env->DeleteLocalRef(result_class);
    return result;
  } catch (const std::exception &e) {
    error(env, e);
    return nullptr;
  }
}
extern "C" JNIEXPORT void JNICALL
Java_ai_gradium_phonon_QnnNative_close(JNIEnv *env, jobject, jlong handle) {
  try {
    delete &session(handle);
  } catch (const std::exception &e) {
    error(env, e);
  }
}
