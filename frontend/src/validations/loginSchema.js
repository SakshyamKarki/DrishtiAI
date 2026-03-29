import * as Yup from 'yup';

const loginSchema = Yup.object({
    // email: Yup
    //     .string()
    //     .email("Please enter a valid email")
    //     .required("Email is required"),
    password: Yup
        .string()
        .min(8, "Password must be at least 8 characters")
        .required("Password is required"),
    username: Yup   
        .string()
        .min(1, "Username cannot be smaller than 1 character")
        .required("Username is required")
});

export default loginSchema;