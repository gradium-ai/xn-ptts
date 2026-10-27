plugins { id("com.android.application"); id("org.jetbrains.kotlin.android") }
android {
    namespace = "ai.gradium.phonon.smoke"
    compileSdk = 36
    defaultConfig {
        applicationId = "ai.gradium.phonon.smoke"
        minSdk = 31
        targetSdk = 36
        ndk { abiFilters += "arm64-v8a" }
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }
    packaging { jniLibs { useLegacyPackaging = true } }
    buildTypes { release { isMinifyEnabled = true; signingConfig = signingConfigs.getByName("debug"); proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt")) } }
    compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
}
kotlin { compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) } }
dependencies {
    // Pass -PpttsRepository=/path/to/repository to test the published AAR and POM instead.
    if (providers.gradleProperty("pttsRepository").isPresent) {
        implementation("ai.gradium:ptts:${providers.gradleProperty("pttsVersion").get()}")
    } else {
        implementation(project(":ptts"))
    }
    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
}
