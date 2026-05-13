import java.util.Properties
import java.io.FileInputStream

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.google.gms.google.services)

    id("kotlin-kapt")
}

// FIXED: Load local.properties at TOP (global scope, before android block)
val localProperties = Properties()
val localPropertiesFile = rootProject.file("local.properties")
if (localPropertiesFile.exists()) {
    localProperties.load(FileInputStream(localPropertiesFile))
}
val geminiApiKey = localProperties.getProperty("GEMINI_API_KEY") ?: ""
val groqApiKey = localProperties.getProperty("GROQ_API_KEY") ?: ""

// Sanitize the API keys: remove surrounding quotes if the user added them
val sanitizedGeminiApiKey = if (geminiApiKey.startsWith("\"") && geminiApiKey.endsWith("\"")) {
    geminiApiKey.substring(1, geminiApiKey.length - 1)
} else {
    geminiApiKey
}
val sanitizedGroqApiKey = if (groqApiKey.startsWith("\"") && groqApiKey.endsWith("\"")) {
    groqApiKey.substring(1, groqApiKey.length - 1)
} else {
    groqApiKey
}

android {
    namespace = "com.fyp.nextshot"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.fyp.nextshot"
        minSdk = 24
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        
        // Define the API Keys in BuildConfig
        buildConfigField("String", "GEMINI_API_KEY", "\"$sanitizedGeminiApiKey\"")
        buildConfigField("String", "GROQ_API_KEY", "\"$sanitizedGroqApiKey\"")
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions {
        jvmTarget = "11"
    }
    buildFeatures {
        viewBinding = true
        buildConfig = true
    }
}

kapt {
    correctErrorTypes = true
}

// Exclude older intellij annotations to prevent duplicate class errors
configurations.all {
    exclude(group = "com.intellij", module = "annotations")
}

dependencies {

    // AndroidX & UI Essentials
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    implementation(libs.androidx.activity)
    implementation(libs.androidx.constraintlayout)

    // Lifecycle
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.7.0")

    // Image Loading (Glide)
    implementation("com.github.bumptech.glide:glide:4.16.0")
    implementation(libs.androidx.credentials)
    implementation(libs.androidx.credentials.play.services.auth)
    implementation(libs.googleid)
    implementation(libs.androidx.ui)
    implementation(libs.firebase.firestore)
    kapt("com.github.bumptech.glide:compiler:4.16.0")
    
    // Google AI Client
    implementation("com.google.ai.client.generativeai:generativeai:0.9.0")
    
    // Guava (for ListenableFuture) - Needed by CameraX
    implementation("com.google.guava:guava:33.0.0-android")

    implementation("com.squareup.okhttp3:okhttp:4.12.0")

    // Navigation
    implementation(libs.androidx.navigation.fragment.ktx)
    implementation(libs.androidx.navigation.ui.ktx)

    // Firebase Bill of Materials (BOM)
    implementation(platform("com.google.firebase:firebase-bom:33.1.2"))

    // Firebase Libraries
    implementation("com.google.firebase:firebase-auth-ktx")
    implementation("com.google.firebase:firebase-firestore-ktx") 
    implementation("com.google.firebase:firebase-storage")
    implementation("com.google.firebase:firebase-messaging")

    // Google Services
    implementation("com.google.android.gms:play-services-auth:20.7.0")
    implementation("com.google.code.gson:gson:2.10.1")

    // Testing
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)

    // Retrofit
    implementation("com.squareup.retrofit2:retrofit:2.9.0")
    implementation("com.squareup.retrofit2:converter-gson:2.9.0")

    // Coroutines
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.7.3")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")

    implementation(libs.androidx.room.runtime)
    implementation(libs.androidx.room.ktx)
    kapt(libs.androidx.room.compiler)
    kapt("org.jetbrains.kotlinx:kotlinx-metadata-jvm:0.9.0")

    // CameraX
    val cameraXVersion = "1.3.1"
    implementation("androidx.camera:camera-core:$cameraXVersion")
    implementation("androidx.camera:camera-camera2:$cameraXVersion")
    implementation("androidx.camera:camera-lifecycle:$cameraXVersion")
    implementation("androidx.camera:camera-view:$cameraXVersion")
    implementation("androidx.camera:camera-video:1.3.4")
    implementation("com.github.PhilJay:MPAndroidChart:v3.1.0")

    // YouTube Player Library
    implementation("com.pierfrancescosoffritti.androidyoutubeplayer:core:12.1.0")
    implementation("com.google.mediapipe:tasks-vision:0.10.14")

    //Mediapipe
    implementation("com.google.mediapipe:tasks-vision:0.10.14")
}
