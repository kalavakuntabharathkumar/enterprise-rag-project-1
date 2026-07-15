/** @type {import('next').NextConfig} */
const nextConfig = {
  // Allow Replit's proxied preview domain to load hot-reload resources.
  allowedDevOrigins: ["*.replit.dev", "*.sisko.replit.dev", "*.repl.co"],
};

export default nextConfig;
