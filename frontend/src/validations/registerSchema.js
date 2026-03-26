import * as Yup from 'yup';

const registerSchema = Yup.object({
    username: Yup
        .string()
        .required("Username is required")
        .max(20, "Character length exceed maximum length"),  
    email: Yup
        .string()
        .email("Please enter a valid email")
        .required("Email is required"),
    password: Yup
        .string()
        .min(8, "Password must be at least 8 characters")
        .required("Password is required"),
    confirmPassword: Yup
        .string()
        .oneOf([Yup.ref("password")], "Passwords do not match")
        .required("Please confirm your password")
});

export default registerSchema;