import { index, route } from "@react-router/dev/routes";

export default [index("./home.js"), route("products/:pid", "./product.tsx")];
