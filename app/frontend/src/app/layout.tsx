import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SAM3 Segmentation Studio",
  description: "Interactive image segmentation with SAM3 model",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="antialiased min-h-screen bg-gradient-to-br from-background via-background to-card">
        {children}
      </body>
    </html>
  );
}

