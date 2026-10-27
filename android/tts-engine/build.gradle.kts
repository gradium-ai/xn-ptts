plugins {
    id("com.android.application") version "8.13.0"
    id("org.jetbrains.kotlin.android") version "2.2.20"
}

val workspaceVersion = Regex("(?m)^version = \"([^\"]+)\"")
    .find(file("../../Cargo.toml").readText())!!.groupValues[1]

// The engine speaks through the same Kotlin wrapper apps copy, android/PhononTTS.kt, so it is
// compiled from there rather than kept as a second copy.
abstract class ShareWrapper : DefaultTask() {
    @get:InputFile abstract val source: RegularFileProperty
    @get:OutputDirectory abstract val output: DirectoryProperty

    @TaskAction
    fun copy() {
        source.get().asFile.copyTo(output.file("PhononTTS.kt").get().asFile, overwrite = true)
    }
}
val shareWrapper = tasks.register<ShareWrapper>("shareWrapper") { source.set(file("../PhononTTS.kt")) }
androidComponents.onVariants { it.sources.java?.addGeneratedSourceDirectory(shareWrapper, ShareWrapper::output) }

android {
    namespace = "ai.gradium.phonon.tts"
    compileSdk = 36
    defaultConfig {
        applicationId = "ai.gradium.phonon.tts"
        minSdk = 24
        targetSdk = 36
        versionCode = 1
        versionName = workspaceVersion
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }
    // What android/build.sh writes.
    sourceSets["main"].jniLibs.srcDir("../jniLibs")
    buildTypes {
        release {
            isMinifyEnabled = true
            // So a release build installs for testing. Sign a published build with your own key.
            signingConfig = signingConfigs.getByName("debug")
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

kotlin { compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) } }

dependencies {
    implementation("net.java.dev.jna:jna:5.17.0@aar")
    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
}
