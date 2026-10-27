pluginManagement { repositories { google(); mavenCentral(); gradlePluginPortal() } }
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        providers.gradleProperty("pttsRepository").orNull?.let { maven { url = uri(it) } }
        google(); mavenCentral()
    }
}
rootProject.name = "ptts-android"
include(":ptts", ":smoke")
