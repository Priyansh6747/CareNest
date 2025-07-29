// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
    apiKey: "AIzaSyCaF6E2Vn4pEzJRqzW9NSsQkR6xgDi-yaw",
    authDomain: "carenest-fde45.firebaseapp.com",
    projectId: "carenest-fde45",
    storageBucket: "carenest-fde45.firebasestorage.app",
    messagingSenderId: "605090892848",
    appId: "1:605090892848:web:875dd79600b4d3fabcb1f5",
    measurementId: "G-ERECZJS55J"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);