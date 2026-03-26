import * as Yup from 'yup';

const loginSchema = Yup.object({
    email: Yup
        .string()
        .email("Please enter a valid email")
        .required("Email is required"),
    password: Yup
        .string()
        .min(8, "Password must be at least 8 characters")
        .required("Password is required"),
});

export default loginSchema;