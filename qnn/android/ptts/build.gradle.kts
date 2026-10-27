plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
    id("maven-publish")
}
val workspaceVersion = Regex("(?m)^version = \"([^\"]+)\"").find(file("../../../Cargo.toml").readText())!!.groupValues[1]
group = "ai.gradium"
version = workspaceVersion
val unitTestsOnly = providers.gradleProperty("pttsUnitTestsOnly").orNull == "true"
if (unitTestsOnly) {
    require(gradle.startParameter.taskNames == listOf(":ptts:testDebugUnitTest")) {
        "pttsUnitTestsOnly is restricted to :ptts:testDebugUnitTest; it cannot build an AAR"
    }
}
val sdk = providers.environmentVariable("QAIRT_ROOT").orNull
    ?: if (unitTestsOnly) "" else error("Set QAIRT_ROOT to the QAIRT 2.50.0 SDK used to compile the model")
android {
    namespace = "ai.gradium.phonon"
    compileSdk = 36
    ndkVersion = "29.0.14206865"
    defaultConfig {
        minSdk = 31
        ndk { abiFilters += "arm64-v8a" }
        consumerProguardFiles("consumer-rules.pro")
        externalNativeBuild { cmake {
            arguments += listOf("-DQAIRT_ROOT=$sdk", "-DANDROID_STL=c++_static")
            targets += "ptts_qnn"
        } }
    }
    if (!unitTestsOnly) externalNativeBuild { cmake { path = file("src/main/cpp/CMakeLists.txt"); version = "3.31.6" } }
    compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
    publishing { singleVariant("release") { withSourcesJar() } }
    testOptions { unitTests.isReturnDefaultValues = true }
}
kotlin { compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) } }
val buildText by tasks.registering(Exec::class) {
    workingDir = file("../../text-ffi")
    commandLine("bash", "build.sh", "android")
    environment("ANDROID_NDK_HOME", android.ndkDirectory.absolutePath)
    inputs.files(fileTree("../../text-ffi/src"), file("../../text-ffi/Cargo.toml"), file("../../text-ffi/Cargo.lock"), file("../../text-ffi/.cargo/config.toml"), fileTree("../../../ptts/src"), file("../../../ptts/Cargo.toml"), file("../../../Cargo.toml"))
    outputs.file("../../text-ffi/dist/android-arm64-v8a/libptts_text.so")
}
val packageText by tasks.registering(Copy::class) {
    dependsOn(buildText)
    from("../../text-ffi/dist/android-arm64-v8a/libptts_text.so")
    into("src/main/jniLibs/arm64-v8a")
}
if (!unitTestsOnly) tasks.named("preBuild") { dependsOn(packageText) }
dependencies {
    implementation("com.qualcomm.qti:qnn-runtime:2.50.0")
    testImplementation("junit:junit:4.13.2")
}
afterEvaluate {
    publishing {
        repositories { maven { name = "verification"; url = uri(layout.buildDirectory.dir("repository")) } }
        publications {
        create<MavenPublication>("release") {
            from(components["release"])
            artifactId = "ptts"
            pom {
                name.set("Phonon TTS")
                description.set("Streaming Phonon speech synthesis on Snapdragon NPUs")
                url.set("https://github.com/gradium-ai/xn-ptts")
                licenses {
                    license { name.set("MIT"); url.set("https://opensource.org/license/mit") }
                    license { name.set("Apache-2.0"); url.set("https://www.apache.org/licenses/LICENSE-2.0") }
                }
                scm { url.set("https://github.com/gradium-ai/xn-ptts") }
            }
        }
    } }
}
